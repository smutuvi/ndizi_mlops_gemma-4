#!/usr/bin/env python3
"""
Evaluate finetuned Gemma 4 LoRA adapter (Hub test splits or prepared artifact) or transcribe one file.

Hub batch eval (ndizi_mlops-style):

  python scripts/evaluate_gemma4.py \\
    --checkpoint artifacts/checkpoints/best \\
    --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \\
    --output_dir eval/gemma4-ndizi-run1 \\
    --chunk_length_s 30 \\
    --batch_size 4 \\
    --normalize jiwer_default

Zero-shot base model (E2B) on one audio file:

  python scripts/evaluate_gemma4.py \\
    --model E2B \\
    --baseline \\
    --audio /path/to/clip.wav \\
    --reference "optional ground truth" \\
    --output eval/baseline_single.json

Long clips (>30s): omit --chunk_length_s to auto-enable 30s windows, or pass explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.finetuned import run_evaluate, run_transcribe_file  # noqa: E402
from src.eval.normalize import TEXT_NORMALIZE_EVAL_DEFAULT, add_normalize_arg  # noqa: E402
from src.utils.constants import ASR_INSTRUCTION, MAX_AUDIO_SEC, PUNCTUATION_ASR_INSTRUCTION  # noqa: E402
from src.utils.paths import CHECKPOINT_DIR  # noqa: E402
from src.utils.runtime import apply_model_choice  # noqa: E402


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="E2B", help="E2B / E4B or full HuggingFace model id")

    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"LoRA adapter directory (default: {CHECKPOINT_DIR / 'best'})",
    )
    p.add_argument(
        "--test_datasets",
        nargs="+",
        default=None,
        help='Hub splits e.g. smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test (default: artifacts/prepared_dataset test)',
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write metrics.json, predictions.json, predictions.csv here (default: artifacts/predictions)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows per Hub split (debug)")
    p.add_argument("--dataset-revision", type=str, default=None)
    p.add_argument("--audio-column", type=str, default=None)
    p.add_argument("--text-column", type=str, default=None)
    p.add_argument("--retention-eval", action="store_true", help="Also eval retention prepared test (local only)")

    p.add_argument(
        "--chunk_length_s",
        "--chunk-long-audio-seconds",
        type=float,
        default=None,
        dest="chunk_length_s",
        metavar="SEC",
        help=f"Split long audio into chunks of SEC seconds (e.g. 30). Auto={MAX_AUDIO_SEC}s if any clip exceeds limit.",
    )
    p.add_argument(
        "--stride_length_s",
        type=float,
        default=None,
        help="Stride between chunks in seconds (default: no overlap, same as chunk length).",
    )
    p.add_argument(
        "--no-auto-chunk",
        action="store_true",
        help="Do not auto-enable chunking for clips longer than Gemma's 30s limit.",
    )
    p.add_argument(
        "--max_audio_seconds",
        type=float,
        default=None,
        help="Drop clips longer than SEC when chunking is disabled.",
    )
    p.add_argument("--fp16", action="store_true", help="Load base model in float16 (default: bfloat16)")
    p.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens")
    p.add_argument("--repetition-penalty", type=float, default=None, dest="repetition_penalty")
    p.add_argument("--no-repeat-ngram-size", type=int, default=None, dest="no_repeat_ngram_size")
    p.add_argument(
        "--anti-loop-decode",
        action="store_true",
        help="Convenience flag: set max_new_tokens=128, repetition_penalty=1.1, no_repeat_ngram_size=4",
    )
    p.add_argument(
        "--aggressive-qc",
        action="store_true",
        help="Optional multi-gate QC filter (off by default; can drop noisy rows before scoring).",
    )

    p.add_argument(
        "--baseline",
        action="store_true",
        help="Zero-shot base model (no LoRA checkpoint). Use with --audio for single-file inference.",
    )
    p.add_argument(
        "--no-adapter",
        action="store_true",
        dest="no_adapter",
        help="Load --model directly without a LoRA adapter. Use for evaluating merged Hub models "
             "(e.g. --model smutuvi/gemma-4-e2b-sw-asr-ndizi-merged --no-adapter).",
    )
    p.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Transcribe a single audio file instead of batch eval",
    )
    p.add_argument("--reference", type=str, default=None, help="Reference text for WER when using --audio")
    p.add_argument("--output", type=str, default=None, help="JSON path for single-file --audio mode")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: cap max_samples to a tiny value and write to output_dir/smoke",
    )

    p.add_argument(
        "--instruction",
        choices=("default", "punctuation"),
        default="punctuation",
        help="ASR instruction variant: 'punctuation' (default) forces punctuation+casing; "
             "'default' uses the softer variant.",
    )
    add_normalize_arg(p, default=TEXT_NORMALIZE_EVAL_DEFAULT)
    args = p.parse_args()
    args.asr_instruction = (
        PUNCTUATION_ASR_INSTRUCTION if args.instruction == "punctuation" else ASR_INSTRUCTION
    )

    if args.anti_loop_decode:
        if args.max_new_tokens == 256:
            args.max_new_tokens = 128
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.1
        if args.no_repeat_ngram_size is None:
            args.no_repeat_ngram_size = 4

    if args.smoke:
        if args.max_samples is None:
            args.max_samples = 8
        if args.output_dir:
            args.output_dir = str(Path(args.output_dir) / "smoke")
        else:
            args.output_dir = "artifacts/predictions/smoke"

    apply_model_choice(args.model)
    if args.audio:
        run_transcribe_file(args)
    elif args.baseline:
        raise SystemExit("--baseline requires --audio (single-file zero-shot inference).")
    else:
        run_evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
