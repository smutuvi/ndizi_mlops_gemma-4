#!/usr/bin/env bash
# Training mode guide:
#   asr_safe     — projectors only, no LM LoRA; preserves chat. Run first.
#                  python scripts/train_gemma4.py --training-mode asr_safe --short-instruction --lr 1e-4 --epochs 1
#   asr_moderate — tail LoRA (last 6 layers) + projectors; balanced.
#                  python scripts/train_gemma4.py --training-mode asr_moderate --short-instruction --lr 5e-5 --epochs 1
#   asr_max      — full decoder LoRA (existing); best ASR, degrades chat. ASR-only bundles only.
#                  python scripts/train_gemma4.py --training-mode asr_max --lr 1e-4 --epochs 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/run_pipeline.py all \
  --chunk-long-audio \
  --retention-eval \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  "$@"
