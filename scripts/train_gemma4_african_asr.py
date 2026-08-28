#!/usr/bin/env python3
"""Train Sunflower-Gemma4 on Ndizi + other Swahili ASR + WaxalNLP Amharic/Oromo.

Uses the Gemma 4 QLoRA path that actually runs (batch size 1, no dtype= with
BitsAndBytes, LM-only LoRA regex, audio_tower left in bf16). Per-row ASR prompts:
Swahili LiteRT string, short Amharic / Oromo prompts.

Hub *test* splits are never loaded into train. FLEURS sw_ke test stays eval-only.

  python scripts/train_gemma4_african_asr.py --prepare

  python scripts/train_gemma4_african_asr.py \\
    --training-mode asr_moderate --asr-prompt ondevice \\
    --output-dir artifacts/checkpoints_african_asr

  # Uncapped Waxal (large download)
  python scripts/train_gemma4_african_asr.py --prepare --full-waxal --train
"""
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

    from src.utils.constants import SUNFLOWER_MODEL_ID, SUNFLOWER_SYSTEM_PROMPT  # noqa: E402
    from src.utils.paths import AFRICAN_ASR_CHECKPOINT_DIR, AFRICAN_ASR_PREPARED_LOCAL  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=SUNFLOWER_MODEL_ID)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument(
        "--extra-asr",
        nargs="*",
        default=[],
        metavar="SPEC",
        help="Extra Hub sources as repo, repo:config, or repo:config:lang (lang=sw|am|om).",
    )
    p.add_argument("--waxal-max", type=int, default=20_000, help="Cap WaxalNLP train rows per language.")
    p.add_argument("--full-waxal", action="store_true", help="Do not cap WaxalNLP train splits.")
    p.add_argument("--sw-prob", type=float, default=0.5)
    p.add_argument("--am-prob", type=float, default=0.25)
    p.add_argument("--om-prob", type=float, default=0.25)
    p.add_argument("--prepared-dir", default=str(AFRICAN_ASR_PREPARED_LOCAL))
    p.add_argument(
        "--training-mode",
        choices=("asr_safe", "asr_moderate", "asr_max"),
        default="asr_moderate",
    )
    p.add_argument("--tail-lora-layers", type=int, default=6)
    p.add_argument("--tail-lora-rank", type=int, default=8)
    p.add_argument(
        "--asr-prompt",
        choices=("ondevice", "short", "full", "punctuation"),
        default="ondevice",
        help="Fallback prompt if a row has no asr_instruction (rows are tagged per language).",
    )
    p.add_argument("--unfreeze-audio-tower", dest="unfreeze_audio_tower", action="store_true")
    p.add_argument("--freeze-audio-tower", dest="unfreeze_audio_tower", action="store_false")
    p.set_defaults(unfreeze_audio_tower=True)
    p.add_argument("--audio-tower-last-layers", type=int, default=4)
    p.add_argument("--chat-jsonl", default=str(ROOT / "data" / "sunflower_chat_train.jsonl"))
    p.add_argument("--chat-ratio", type=float, default=0.1)
    p.add_argument("--no-chat-mix", action="store_true")
    p.add_argument("--system-prompt", default=SUNFLOWER_SYSTEM_PROMPT)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler", type=str, default="cosine")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-max-samples", type=int, default=64)
    p.add_argument("--no-train-eval", action="store_true")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--lora-target-modules", default=None)
    p.add_argument("--debug-lora-targets", action="store_true")
    p.add_argument("--peft-clippable-patch", action="store_true")
    p.add_argument("--output-dir", default=str(AFRICAN_ASR_CHECKPOINT_DIR))
    args = p.parse_args()

    do_train = args.train or not args.prepare
    if args.no_chat_mix:
        args.chat_ratio = 0.0
    chat_path = Path(args.chat_jsonl)
    if args.chat_ratio > 0 and not chat_path.is_file():
        print(f"[warn] --chat-jsonl missing ({chat_path}); training ASR only.")
        args.chat_ratio = 0.0
        args.chat_jsonl = None

    from huggingface_hub import login as hf_login

    from src.utils.runtime import apply_model_choice

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if tok:
        hf_login(token=tok)

    apply_model_choice(args.model)

    if args.prepare or not Path(args.prepared_dir).exists():
        from src.data.prepare_african_asr import run_prepare_african_asr

        run_prepare_african_asr(args)
        if not do_train:
            return 0

    if not Path(args.prepared_dir).exists():
        raise SystemExit(f"Prepared dataset missing: {args.prepared_dir}\nRun with --prepare first.")

    from src.training.train import run_train

    run_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
