#!/usr/bin/env python3
"""Zero-shot Gemma 4 baseline WER/CER (+ optional Whisper reference)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.baseline import run_baseline  # noqa: E402
from src.eval.normalize import add_normalize_arg  # noqa: E402
from src.utils.runtime import apply_model_choice  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="E2B")
    p.add_argument("--with-whisper", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--retention-eval", action="store_true")
    add_normalize_arg(p)
    args = p.parse_args()
    apply_model_choice(args.model)
    run_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
