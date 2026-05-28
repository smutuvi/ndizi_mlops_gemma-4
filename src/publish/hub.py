# src/publish/hub.py — push adapter or gated merged weights to the Hugging Face Hub.
from __future__ import annotations

import json
import textwrap
from typing import Any

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.utils.constants import SRC_DATASETS
from src.utils.paths import (
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


def run_publish(args) -> None:
    rt = get_runtime()
    api = HfApi()
    api.create_repo(rt.output_model_repo, repo_type="model", private=False, exist_ok=True)

    processor = AutoProcessor.from_pretrained(rt.base_model_id)

    card = textwrap.dedent(
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
    """
    ).strip()

    if args.merged:
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
        merged = PeftModel.from_pretrained(base, str(CHECKPOINT_DIR / "best"))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(MERGED_LOCAL), safe_serialization=True)
        processor.save_pretrained(str(MERGED_LOCAL))
        (MERGED_LOCAL / "README.md").write_text(card, encoding="utf-8")
        api.upload_folder(repo_id=rt.output_model_repo, folder_path=str(MERGED_LOCAL), repo_type="model")
    else:
        adapter_dir = CHECKPOINT_DIR / "best"
        (adapter_dir / "README.md").write_text(card, encoding="utf-8")
        api.upload_folder(repo_id=rt.output_model_repo, folder_path=str(adapter_dir), repo_type="model")
    print(f"Published to https://huggingface.co/{rt.output_model_repo}")
