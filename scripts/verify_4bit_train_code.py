#!/usr/bin/env python3
"""Verify this checkout has 4-bit Gemma4 audio training patches (commit 323f948+)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src/training/train.py"
LORA = ROOT / "src/models/gemma4_lora.py"

REQUIRED_TRAIN_MARKERS = (
    "patch_gemma4_audio_finfo_for_kbit",
    "patch_gemma4_masked_scatter_dtype",
    "NDIZI_TRAIN_CODE_VERSION",
)
REQUIRED_LORA_MARKERS = (
    "_gradient_clip_cap",
    "patch_gemma4_audio_finfo_for_kbit",
    "Gemma4AudioFeedForward",
)


def main() -> int:
    ok = True
    for path, markers in ((TRAIN, REQUIRED_TRAIN_MARKERS), (LORA, REQUIRED_LORA_MARKERS)):
        if not path.is_file():
            print(f"MISSING: {path}")
            ok = False
            continue
        text = path.read_text(encoding="utf-8")
        for m in markers:
            if m not in text:
                print(f"MISSING in {path.name}: {m}")
                ok = False
    if ok:
        nlines = len(TRAIN.read_text(encoding="utf-8").splitlines())
        print("OK: 4-bit train patches present.")
        print(f"  {TRAIN.name}: {nlines} lines (expect ~240; stale tree is ~196)")
        return 0
    print("\nFix: cd", ROOT, "&& git fetch origin && git reset --hard origin/main")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
