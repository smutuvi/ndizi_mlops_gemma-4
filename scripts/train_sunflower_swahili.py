#!/usr/bin/env python3
"""Leakage-safe Swahili SFT of Sunbird Sunflower-Gemma4-E2B on Ndizi ASR.

Train only on smutuvi/ndizi-1 + smutuvi/ndizi-1-2025 (optional de-duped ALFFA).
Never trains on FLEURS, Common Voice, SALT, Waxal, or Sunbird/speech.

  # 1. Prepare (anti-join FLEURS + SALT transcripts)
  python scripts/train_sunflower_swahili.py --prepare --chunk-long-audio

  # 2. Optional ALFFA news ASR after de-dup
  python scripts/train_sunflower_swahili.py --prepare --extra-asr nickdee96/ALFFA-Swahili-News

  # 3. Train (asr_moderate preserves chat better than asr_max)
  python scripts/train_sunflower_swahili.py \\
    --training-mode asr_moderate --short-instruction \\
    --chat-jsonl data/sunflower_chat_train.jsonl --chat-ratio 0.2

  Then evaluate with scripts/evaluate_sunflower_swahili.py
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

    from src.utils.constants import (  # noqa: E402
        SUNFLOWER_MODEL_ID,
        SUNFLOWER_OPTIONAL_ASR,
        SUNFLOWER_SYSTEM_PROMPT,
        SUNFLOWER_TRAIN_ASR,
    )
    from src.utils.paths import SUNFLOWER_CHECKPOINT_DIR, SUNFLOWER_PREPARED_LOCAL  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default=SUNFLOWER_MODEL_ID,
        help=f"Base multimodal model (default: {SUNFLOWER_MODEL_ID})",
    )
    p.add_argument("--prepare", action="store_true", help="Build leakage-safe dataset then exit unless --train")
    p.add_argument("--train", action="store_true", help="Run training (implied if --prepare is omitted)")
    p.add_argument("--train-datasets", nargs="+", default=list(SUNFLOWER_TRAIN_ASR))
    p.add_argument(
        "--extra-asr",
        nargs="*",
        default=[],
        metavar="REPO",
        help=f"Optional extra ASR Hub repos after leakage filter (e.g. {SUNFLOWER_OPTIONAL_ASR})",
    )
    p.add_argument("--skip-leakage-filter", action="store_true")
    p.add_argument("--skip-fleurs-block", action="store_true")
    p.add_argument("--skip-salt-block", action="store_true")
    p.add_argument("--chunk-long-audio", action="store_true")
    p.add_argument("--chunk-test", action="store_true")
    p.add_argument("--prepared-dir", default=str(SUNFLOWER_PREPARED_LOCAL))
    p.add_argument(
        "--training-mode",
        choices=("asr_safe", "asr_moderate", "asr_max"),
        default="asr_moderate",
        help="asr_moderate (default) = tail LoRA; asr_max tends to wipe Sunflower chat.",
    )
    p.add_argument("--tail-lora-layers", type=int, default=6)
    p.add_argument("--tail-lora-rank", type=int, default=8)
    p.add_argument(
        "--asr-prompt",
        choices=("ondevice", "short", "full", "punctuation"),
        default="ondevice",
        help="Train with the LiteRT on-device Swahili prompt (default). "
        "short/full/punctuation are English eval-style prompts.",
    )
    p.add_argument(
        "--short-instruction",
        dest="asr_prompt",
        action="store_const",
        const="short",
        help="Alias for --asr-prompt short.",
    )
    p.add_argument(
        "--full-instruction",
        dest="asr_prompt",
        action="store_const",
        const="full",
        help="Alias for --asr-prompt full.",
    )
    p.add_argument(
        "--unfreeze-audio-tower",
        dest="unfreeze_audio_tower",
        action="store_true",
        help="Train last audio_tower layers (needed for noisy Ndizi; default on).",
    )
    p.add_argument(
        "--freeze-audio-tower",
        dest="unfreeze_audio_tower",
        action="store_false",
        help="Keep the audio encoder frozen (embed_audio only).",
    )
    p.set_defaults(unfreeze_audio_tower=True)
    p.add_argument(
        "--audio-tower-last-layers",
        type=int,
        default=4,
        help="How many trailing audio_tower layers to train (0 = all). Default 4.",
    )
    p.add_argument("--chat-jsonl", default=str(ROOT / "data" / "sunflower_chat_train.jsonl"))
    p.add_argument("--chat-ratio", type=float, default=0.2, help="Mix ratio for text chat rows (0 disables).")
    p.add_argument("--no-chat-mix", action="store_true")
    p.add_argument("--system-prompt", default=SUNFLOWER_SYSTEM_PROMPT)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=2.0)
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
    p.add_argument("--output-dir", default=str(SUNFLOWER_CHECKPOINT_DIR))
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
        from src.data.prepare_sunflower import run_prepare_sunflower

        prep = argparse.Namespace(
            train_datasets=args.train_datasets,
            extra_asr=args.extra_asr,
            skip_leakage_filter=args.skip_leakage_filter,
            skip_fleurs_block=args.skip_fleurs_block,
            skip_salt_block=args.skip_salt_block,
            chunk_long_audio=args.chunk_long_audio,
            chunk_test=args.chunk_test,
            output_dir=args.prepared_dir,
        )
        run_prepare_sunflower(prep)
        if not do_train:
            return 0

    if not Path(args.prepared_dir).exists():
        raise SystemExit(f"Prepared dataset missing: {args.prepared_dir}\nRun with --prepare first.")

    from src.training.train import run_train

    run_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
