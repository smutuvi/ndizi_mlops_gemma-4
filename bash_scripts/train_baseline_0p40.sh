#!/usr/bin/env bash
# Train command for the ~0.41 normalized WER run (see configs/baseline/e2b_wer_0.41_recipe.yaml).
# Prerequisite: python scripts/prepare_gemma4.py --chunk-long-audio --chunk-test
# Requires git main @ 323f948+ (4-bit audio-tower patches in src/models/gemma4_lora.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! grep -q patch_gemma4_audio_finfo_for_kbit src/training/train.py 2>/dev/null; then
  echo "ERROR: stale train.py (missing 4-bit audio patches). Run: git pull origin main" >&2
  exit 1
fi

python scripts/train_gemma4.py \
  --model E2B \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  --epochs 2 \
  --grad-accum 16 \
  --eval-max-samples 64 \
  --save-steps 500
