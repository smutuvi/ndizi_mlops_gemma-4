#!/usr/bin/env bash
# Restore the training/eval recipe that reached ~0.41 normalized pooled WER (raw ~0.45).
# Does NOT use --aggressive-qc on prepare (QC shrinks/changes train data and hurt this run).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1) Optional: back up current checkpoint before overwrite ==="
if [[ -d artifacts/checkpoints/best ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  cp -a artifacts/checkpoints/best "artifacts/checkpoints/best-before-restore-${STAMP}"
  echo "Backed up to artifacts/checkpoints/best-before-restore-${STAMP}"
fi

echo "=== 2) Re-prepare WITHOUT aggressive QC (chunk long test clips only) ==="
python scripts/prepare_gemma4.py \
  --chunk-long-audio \
  --chunk-test

echo "=== 3) Train (same recipe as 0.40 WER eval run) ==="
python scripts/train_gemma4.py \
  --model E2B \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  --epochs 2 \
  --grad-accum 16 \
  --eval-max-samples 64 \
  --save-steps 500

echo "=== 4) Eval on Hub test (chunked + jiwer_default + anti-loop) ==="
python scripts/evaluate_gemma4.py \
  --model E2B \
  --checkpoint artifacts/checkpoints/best \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-baseline-0p40-restore \
  --chunk_length_s 30 \
  --normalize jiwer_default \
  --anti-loop-decode

echo "Done. See eval/gemma4-baseline-0p40-restore/metrics.json"
