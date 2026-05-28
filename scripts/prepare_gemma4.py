#!/usr/bin/env python3
"""Prepare merged Ndizi datasets (+ optional retention suite) for Gemma 4 ASR."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prepare import run_prepare  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--push", action="store_true")
    p.add_argument("--retention-datasets", nargs="+", default=[])
    p.add_argument("--retention-chunk-test", action="store_true")
    p.add_argument("--chunk-long-audio", action="store_true")
    p.add_argument("--chunk-test", action="store_true")
    p.add_argument(
        "--aggressive-qc",
        action="store_true",
        help="Multi-gate QC filter on each split before merge (off by default). "
        "Use with --chunk-long-audio for long clips.",
    )
    p.add_argument(
        "--qc-use-may6-text-norm",
        action="store_true",
        help="When using --aggressive-qc, apply simple lower+whitespace norm before text gates.",
    )
    p.add_argument(
        "--qc-chunk-long-with-mms-fa",
        action="store_true",
        help="With --aggressive-qc, bump max_dur for QC after MMS-FA chunking (train/val).",
    )
    p.add_argument(
        "--qc-chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk length used for qc max_dur bump when --qc-chunk-long-with-mms-fa.",
    )
    run_prepare(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
