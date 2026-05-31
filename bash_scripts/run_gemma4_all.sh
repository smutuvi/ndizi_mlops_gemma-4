#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/run_pipeline.py all \
  --chunk-long-audio \
  --retention-eval \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  "$@"
