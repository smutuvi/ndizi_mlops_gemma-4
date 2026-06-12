# src/publish/hub.py — push adapter or gated merged weights to the Hugging Face Hub.
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.models.gemma4_lora import (
    is_projector_only_checkpoint,
    load_gemma4_peft_adapter,
    load_projector_checkpoint,
    rewrite_adapter_config_for_kv_shared,
)
from src.utils.constants import SRC_DATASETS
from src.utils.paths import (
    ARTIFACTS_DIR,
    BASELINE_JSON,
    CHECKPOINT_DIR,
    FINETUNED_JSON,
    MERGED_LOCAL,
    RETENTION_BASELINE_JSON,
    RETENTION_FINETUNED_JSON,
)
from src.utils.runtime import get_runtime


def _load_retention_scores() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not (RETENTION_BASELINE_JSON.exists() and RETENTION_FINETUNED_JSON.exists()):
        return None, None
    try:
        b = json.loads(RETENTION_BASELINE_JSON.read_text(encoding="utf-8"))
        f = json.loads(RETENTION_FINETUNED_JSON.read_text(encoding="utf-8"))
        return b, f
    except Exception:
        return None, None


def retention_ok(max_wer_delta: float) -> tuple[bool, str]:
    base_d, ft_d = _load_retention_scores()
    if base_d is None or ft_d is None:
        return False, "Missing retention baseline/finetuned scores (run baseline + evaluate with retention suite)."
    try:
        b = base_d.get("gemma4_retention_test", {}).get("pooled", {}).get("wer")
        t = ft_d.get("retention_test", {}).get("pooled", {}).get("wer")
        if b is None or t is None:
            return False, "Retention scores missing pooled WER."
        delta = float(t) - float(b)
        if delta <= float(max_wer_delta):
            return (
                True,
                f"Retention gate pass: baseline WER={b:.4f}, tuned WER={t:.4f}, "
                f"delta={delta:.4f} <= {max_wer_delta:.4f}",
            )
        return (
            False,
            f"Retention gate FAIL: baseline WER={b:.4f}, tuned WER={t:.4f}, "
            f"delta={delta:.4f} > {max_wer_delta:.4f}",
        )
    except Exception as e:
        return False, f"Retention gate error: {e}"


def wer_table() -> str:
    if not (BASELINE_JSON.exists() and FINETUNED_JSON.exists()):
        return "_(WER table unavailable; run baseline and evaluate first.)_"
    base_d = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    ft_d = json.loads(FINETUNED_JSON.read_text(encoding="utf-8"))
    rows = [
        "| split | group | base WER | tuned WER | base CER | tuned CER | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in sorted(ft_d):
        base_groups = base_d.get(f"gemma4_{split}", {})
        tuned_groups = ft_d[split]
        for group in sorted(tuned_groups, key=lambda x: (x != "pooled", x)):
            b = base_groups.get(group, {})
            t = tuned_groups[group]
            rows.append(
                f"| {split} | {group} | "
                f"{b.get('wer', float('nan')):.3f} | {t['wer']:.3f} | "
                f"{b.get('cer', float('nan')):.3f} | {t['cer']:.3f} | "
                f"{t['n']} |"
            )
    return "\n".join(rows)


def chat_gate_ok(
    adapter_dir: Path,
    prompts_file: Path | None,
    threshold: float,
    *,
    skip: bool = False,
) -> tuple[bool, str]:
    """Run eval_chat.py against the checkpoint and return (passed, message)."""
    if skip:
        return True, "Chat gate skipped (--skip-chat-gate)"

    # Locate eval_chat.py — lives in additional_scripts/ or a sibling scripts dir.
    candidates = [
        Path(__file__).resolve().parents[2] / "additional_scripts" / "eval_chat.py",
        Path(__file__).resolve().parents[2] / "scripts" / "eval_chat.py",
    ]
    eval_chat = next((p for p in candidates if p.exists()), None)
    if eval_chat is None:
        return False, "eval_chat.py not found — copy it to additional_scripts/ or scripts/"

    out_dir = ARTIFACTS_DIR / "chat_gate"
    cmd = [
        sys.executable, str(eval_chat),
        "--model-id", str(adapter_dir),
        "--base-model-id", get_runtime().base_model_id,
        "--output-dir", str(out_dir),
        "--threshold", str(threshold),
        "--device-map", "auto",
    ]
    if prompts_file and prompts_file.exists():
        cmd += ["--prompts-file", str(prompts_file)]

    print("[publish] Running chat gate:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        return True, f"Chat gate PASS (threshold={threshold:.0%})"
    try:
        metrics = json.loads((out_dir / "chat_metrics.json").read_text())
        return False, (
            f"Chat gate FAIL: pass_rate={metrics.get('pass_rate', '?'):.1%} "
            f"< threshold={threshold:.0%}"
        )
    except Exception:
        return False, f"Chat gate FAIL (eval_chat.py exited {result.returncode})"


def run_publish(args) -> None:
    rt = get_runtime()
    api = HfApi()
    publish_merged = bool(args.merged)
    target_repo = rt.merged_model_repo if publish_merged else rt.output_model_repo
    api.create_repo(target_repo, repo_type="model", private=False, exist_ok=True)

    processor = AutoProcessor.from_pretrained(rt.base_model_id)

    adapter_card = textwrap.dedent(
        f"""
    ---
    base_model: {rt.base_model_id}
    library_name: peft
    license: gemma
    language: [sw]
    tags: [automatic-speech-recognition, swahili, gemma-4, lora]
    datasets: [{SRC_DATASETS[0]}, {SRC_DATASETS[1]}]
    ---

    # Gemma 4 - Swahili ASR (ndizi)

    LoRA fine-tune of `{rt.base_model_id}` for Swahili ASR on
    `{SRC_DATASETS[0]}` + `{SRC_DATASETS[1]}` (concatenated).

    ## Evaluation
    {wer_table()}

    ## Training
    QLoRA (4-bit nf4, bf16 compute), rank 32 alpha 64, audio projector unfrozen.

  Merged weights: [{rt.merged_model_repo}](https://huggingface.co/{rt.merged_model_repo})
    """
    ).strip()

    merged_card = textwrap.dedent(
        f"""
    ---
    base_model: {rt.base_model_id}
    license: gemma
    language: [sw]
    tags: [automatic-speech-recognition, swahili, gemma-4]
    datasets: [{SRC_DATASETS[0]}, {SRC_DATASETS[1]}]
    ---

    # Gemma 4 - Swahili ASR (ndizi, merged)

    Full weights: `{rt.base_model_id}` + LoRA from [{rt.output_model_repo}](https://huggingface.co/{rt.output_model_repo}).

    ## Evaluation
    {wer_table()}

    ## Training
    QLoRA (4-bit nf4, bf16 compute), rank 32 alpha 64, audio projector unfrozen.
    """
    ).strip()

    adapter_dir = CHECKPOINT_DIR / "best"

    # Chat gate — runs before any merge/upload.
    skip_chat = bool(getattr(args, "skip_chat_gate", False))
    chat_threshold = float(getattr(args, "chat_gate_threshold", 0.80))
    chat_prompts = getattr(args, "chat_prompts_file", None)
    chat_ok, chat_msg = chat_gate_ok(
        adapter_dir,
        Path(chat_prompts) if chat_prompts else None,
        chat_threshold,
        skip=skip_chat,
    )
    print("[publish] chat gate:", chat_msg)
    if not chat_ok and not bool(getattr(args, "force_merged", False)):
        raise SystemExit(
            "Refusing to publish: chat quality gate failed. "
            "Use --skip-chat-gate or --force-merged to override."
        )

    if publish_merged:
        max_delta = float(getattr(args, "max_retention_wer_delta", 0.02))
        ok, msg = retention_ok(max_delta)
        print("[publish] retention check:", msg)
        if not ok and not bool(getattr(args, "force_merged", False)):
            raise SystemExit(
                "Refusing to publish merged weights due to retention gate failure. "
                "Re-run with better hyperparams / more replay, or pass --force-merged to override."
            )
        base = AutoModelForMultimodalLM.from_pretrained(
            rt.base_model_id, dtype=torch.bfloat16, device_map="cpu"
        )
        if is_projector_only_checkpoint(adapter_dir):
            print("[publish] asr_safe checkpoint detected — loading projector weights onto base model")
            merged = load_projector_checkpoint(base, adapter_dir)
        else:
            merged = load_gemma4_peft_adapter(base, str(adapter_dir))
            merged = merged.merge_and_unload()
        merged.save_pretrained(str(MERGED_LOCAL), safe_serialization=True)
        processor.save_pretrained(str(MERGED_LOCAL))
        (MERGED_LOCAL / "README.md").write_text(merged_card, encoding="utf-8")
        commit = getattr(args, "commit_message", None) or "Publish merged weights from training checkpoint"
        api.upload_folder(
            repo_id=target_repo,
            folder_path=str(MERGED_LOCAL),
            repo_type="model",
            commit_message=commit,
        )
    else:
        if not adapter_dir.is_dir():
            raise SystemExit(f"Adapter checkpoint not found: {adapter_dir}")
        (adapter_dir / "README.md").write_text(adapter_card, encoding="utf-8")
        commit = getattr(args, "commit_message", None) or "Publish LoRA adapter from training checkpoint"
        api.upload_folder(
            repo_id=target_repo,
            folder_path=str(adapter_dir),
            repo_type="model",
            commit_message=commit,
        )
    print(f"Published to https://huggingface.co/{target_repo}")
