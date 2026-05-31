#!/usr/bin/env python3
"""QLoRA fine-tune Gemma 4 for Swahili ASR (adapter-first, optional retention replay)."""
from __future__ import annotations

import argparse
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


def main() -> int:
    load_env_file(ROOT / ".env")
    os.chdir(ROOT)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="E2B")
    p.add_argument("--retention-datasets", nargs="+", default=[])
    p.add_argument("--replay-ratio", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler", type=str, default="cosine")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=64,
        help="Max validation rows per eval step (default 64; full 500 logits can OOM).",
    )
    p.add_argument(
        "--no-train-eval",
        action="store_true",
        help="Disable mid-training eval (avoids eval OOM); save by save_steps only.",
    )
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load bf16 weights instead of 4-bit QLoRA (more VRAM; use if PEFT fails on ClippableLinear).",
    )
    p.add_argument(
        "--lora-target-modules",
        default=None,
        help="Override LoRA target_modules (regex string). Default scopes to language_model only.",
    )
    p.add_argument("--debug-lora-targets", action="store_true", help="Log modules matched by LoRA regex.")
    p.add_argument(
        "--peft-clippable-patch",
        action="store_true",
        help="Monkey-patch Gemma4ClippableLinear before load (last resort; may break 4-bit).",
    )
    args = p.parse_args()

    from huggingface_hub import login as hf_login

    from src.training.train import run_train
    from src.utils.runtime import apply_model_choice

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if tok:
        hf_login(token=tok)

    apply_model_choice(args.model)
    run_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
