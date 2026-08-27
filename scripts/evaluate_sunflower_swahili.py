#!/usr/bin/env python3
"""Evaluate Sunflower-Gemma4-E2B (base or Ndizi LoRA) — ASR + Swahili chat.

ASR eval sets are held-out: Ndizi test (in-domain) and FLEURS sw_ke test (OOD).
FLEURS / SALT / Common Voice / Waxal are never used as training data here.

  python scripts/evaluate_sunflower_swahili.py \\
    --checkpoint artifacts/checkpoints_sunflower_ndizi/best \\
    --output-dir eval/sunflower-ndizi

  # Base Sunflower, no adapter (pre-finetune baseline)
  python scripts/evaluate_sunflower_swahili.py --no-adapter --max-samples 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _chat_once(model, processor, prompt: str, system_prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]
    inputs = processor(text=text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "is_floating_point") and v.is_floating_point():
            inputs[k] = v.to(dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0].strip()


def _chat_ok(prompt: str, reply: str) -> tuple[bool, str]:
    t = (reply or "").strip()
    if not t:
        return False, "empty"
    if t.casefold() == prompt.strip().casefold():
        return False, "echo"
    if len(t.split()) < 3:
        return False, "too_short"
    return True, "ok"


def load_eval_model(args):
    from src.eval.baseline import load_baseline_gemma
    from src.eval.finetuned import load_finetuned_gemma

    if args.no_adapter:
        model, processor = load_baseline_gemma(fp16=args.fp16)
        return model, processor
    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(
            f"LoRA checkpoint not found: {ckpt}\n"
            "Train first, or pass --no-adapter to score the base Sunflower weights."
        )
    model, processor, _ = load_finetuned_gemma(ckpt, fp16=args.fp16)
    return model, processor


def run_chat_eval(args, model, processor, out_dir: Path) -> dict:
    prompts_path = Path(args.chat_prompts)
    prompts: list[dict] = []
    if prompts_path.is_file():
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                if isinstance(rec, str):
                    rec = {"prompt": rec}
                prompts.append(rec)
    for extra in args.chat_prompt_list or []:
        prompts.append({"prompt": extra})

    rows = []
    n_pass = 0
    print(f"\n{'═'*60}\nCHAT EVAL  system={args.system_prompt!r}\n{'═'*60}")
    for i, rec in enumerate(prompts, 1):
        prompt = rec.get("prompt") or rec.get("user") or ""
        reply = _chat_once(model, processor, prompt, args.system_prompt, args.max_new_tokens)
        ok, reason = _chat_ok(prompt, reply)
        n_pass += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  [{i}] {mark} {prompt}\n       {reply[:240]}")
        rows.append({"prompt": prompt, "reply": reply, "ok": ok, "reason": reason})

    n = max(len(rows), 1)
    summary = {"n": len(rows), "n_pass": n_pass, "pass_rate": n_pass / n, "rows": rows}
    (out_dir / "chat.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Chat pass_rate={summary['pass_rate']:.2f}  ({n_pass}/{len(rows)})")
    print("  Wrote", out_dir / "chat.json")
    return summary


def main() -> int:
    load_env_file(ROOT / ".env")
    os.chdir(ROOT)

    from src.eval.finetuned import run_evaluate
    from src.eval.normalize import TEXT_NORMALIZE_EVAL_DEFAULT, add_normalize_arg
    from src.utils.constants import (
        PUNCTUATION_ASR_INSTRUCTION,
        SUNFLOWER_EVAL_ASR,
        SUNFLOWER_MODEL_ID,
        SUNFLOWER_SYSTEM_PROMPT,
    )
    from src.utils.paths import SUNFLOWER_CHECKPOINT_DIR
    from src.utils.runtime import apply_model_choice

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=SUNFLOWER_MODEL_ID, help="Sunflower (or other) multimodal base")
    p.add_argument("--checkpoint", default=str(SUNFLOWER_CHECKPOINT_DIR / "best"))
    p.add_argument("--no-adapter", action="store_true", help="Evaluate the base Sunflower weights only")
    p.add_argument(
        "--test-datasets",
        nargs="+",
        default=list(SUNFLOWER_EVAL_ASR),
        help="Hub specs: repo:split or repo:config:split. Default Ndizi test + FLEURS sw_ke test.",
    )
    p.add_argument("--no-asr", action="store_true", help="Skip ASR WER (chat only)")
    p.add_argument("--no-chat", action="store_true", help="Skip chat prompts (ASR only). Use if VRAM is tight.")
    p.add_argument("--audio-column", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--chat-prompts", default=str(ROOT / "data" / "sunflower_chat_eval.jsonl"))
    p.add_argument("--chat-prompt-list", nargs="+", default=None)
    p.add_argument("--system-prompt", default=SUNFLOWER_SYSTEM_PROMPT)
    p.add_argument("--output-dir", default="eval/sunflower-ndizi")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--chunk_length_s", type=float, default=30.0)
    add_normalize_arg(p, default=TEXT_NORMALIZE_EVAL_DEFAULT)
    args = p.parse_args()
    args.asr_instruction = PUNCTUATION_ASR_INSTRUCTION
    args.no_auto_chunk = False
    args.stride_length_s = None
    args.max_audio_seconds = None
    args.retention_eval = False
    args.baseline = False

    apply_model_choice(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_asr and args.no_chat:
        raise SystemExit("Nothing to run: both --no-asr and --no-chat")
    if not args.no_adapter and not Path(args.checkpoint).is_dir():
        raise SystemExit(
            f"LoRA checkpoint not found: {args.checkpoint}\n"
            "Train first, or pass --no-adapter to score the base Sunflower weights."
        )

    if not args.no_asr:
        run_evaluate(args)

    chat_summary = None
    if not args.no_chat:
        model, processor = load_eval_model(args)
        chat_summary = run_chat_eval(args, model, processor, out_dir)

    payload = {
        "model": args.model,
        "checkpoint": None if args.no_adapter else args.checkpoint,
        "test_datasets": args.test_datasets,
        "chat": {k: chat_summary[k] for k in ("n", "n_pass", "pass_rate")} if chat_summary else None,
    }
    metrics_file = out_dir / "metrics.json"
    if metrics_file.exists():
        payload["asr_metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", summary_path)

    if chat_summary is not None and chat_summary["pass_rate"] < 0.5:
        print("[warn] chat pass_rate < 0.5 — consider asr_moderate, lower merge scale, or more --chat-ratio")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
