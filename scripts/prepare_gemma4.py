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
    run_prepare(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
