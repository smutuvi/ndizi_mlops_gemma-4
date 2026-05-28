#!/usr/bin/env python3
"""Publish Gemma 4 LoRA adapter (default) or gated merged weights to the Hub."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.publish.hub import run_publish  # noqa: E402
from src.utils.runtime import apply_model_choice  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="E2B")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter-only", dest="merged", action="store_false")
    g.add_argument("--merged", dest="merged", action="store_true")
    p.add_argument("--max-retention-wer-delta", type=float, default=0.02)
    p.add_argument("--force-merged", action="store_true")
    args = p.parse_args()
    apply_model_choice(args.model)
    run_publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
