#!/usr/bin/env bash
# Re-run prepare + train for the ~0.41 normalized WER recipe (no aggressive QC on prepare).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 0) Git: use main with 4-bit train fixes (323f948+) ==="
git log -1 --oneline

echo "=== 1) Optional: back up current checkpoint before overwrite ==="
if [[ -d artifacts/checkpoints/best ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  cp -a artifacts/checkpoints/best "artifacts/checkpoints/best-before-restore-${STAMP}"
  echo "Backed up to artifacts/checkpoints/best-before-restore-${STAMP}"
fi

echo "=== 2) Prepare: chunk only (NO --aggressive-qc) ==="
python scripts/prepare_gemma4.py \
  --chunk-long-audio \
  --chunk-test

echo "=== 3) Train (recorded 0.41 WER recipe) ==="
python scripts/train_gemma4.py \
  --model E2B \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  --epochs 2 \
  --grad-accum 16 \
  --eval-max-samples 64 \
  --save-steps 500

echo "=== 4) Eval (exact scoring flags from metrics.json) ==="
bash bash_scripts/eval_baseline_0p40_exact.sh

echo "Done. Compare eval/gemma4-eval-run-chuncked/metrics.json to configs/baseline/metrics_reference.json"
