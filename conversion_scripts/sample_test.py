"""
sample_test.py
──────────────
Quick sanity-check: load a LoRA checkpoint and transcribe N samples
from one or more HuggingFace test splits.  Uses chunked inference so
long clips (>30s) are handled correctly.

Usage:
  python conversion_scripts/sample_test.py
  python conversion_scripts/sample_test.py --checkpoint runs/my-run/best
  python conversion_scripts/sample_test.py --n 5 --datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test
  python conversion_scripts/sample_test.py --audio /path/to/file.wav   # single file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best"
DEFAULT_DATASETS = ["smutuvi/ndizi-1:test", "smutuvi/ndizi-1-2025:test"]
DEFAULT_N = 3
CHUNK_S = 30.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="LoRA adapter directory (default: artifacts/checkpoints/best)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="HF dataset splits to sample from (format: repo:split)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="Number of samples per dataset (default: 3)")
    ap.add_argument("--audio", default=None,
                    help="Single audio file to transcribe (skips dataset sampling)")
    ap.add_argument("--instruction", choices=("default", "punctuation"), default="default")
    args = ap.parse_args()

    import torch
    import datasets as hf_datasets
    from src.eval.finetuned import load_finetuned_gemma
    from src.inference.chunked_transcribe import gemma_transcribe_chunked
    from src.inference.gemma_inputs import load_audio_file
    from src.utils.constants import ASR_INSTRUCTION, PUNCTUATION_ASR_INSTRUCTION

    instruction = ASR_INSTRUCTION if args.instruction == "default" else PUNCTUATION_ASR_INSTRUCTION

    checkpoint = Path(args.checkpoint).expanduser()
    print(f"Checkpoint : {checkpoint.resolve()}")
    print(f"Instruction: {args.instruction}")

    model, processor, _ = load_finetuned_gemma(checkpoint)

    def transcribe(audios):
        return gemma_transcribe_chunked(
            model, processor, audios,
            chunk_length_s=CHUNK_S,
            instruction=instruction,
        )

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.audio:
        audio = load_audio_file(args.audio)
        hyp = transcribe([audio])[0]
        print(f"\n── {Path(args.audio).name} ──")
        print(f"  HYP : {hyp}")
        return 0

    # ── Dataset sampling mode ─────────────────────────────────────────────────
    for spec in args.datasets:
        repo, _, split = spec.partition(":")
        split = split or "test"
        print(f"\n{'═'*60}")
        print(f"Dataset : {repo}  split={split}  n={args.n}")
        print(f"{'═'*60}")

        ds = hf_datasets.load_dataset(repo, split=f"{split}[:{args.n}]")
        for i, row in enumerate(ds):
            hyp = transcribe([row["audio"]])[0]
            ref = row.get("text", row.get("sentence", ""))
            print(f"\n  Sample {i + 1}")
            print(f"  REF : {ref}")
            print(f"  HYP : {hyp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
