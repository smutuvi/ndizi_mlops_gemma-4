#!/usr/bin/env bash
# Re-score artifacts/checkpoints/best with the EXACT flags from predictions/metrics.json.
# Pooled wer_normalized target: 0.40982788132369724 (n=1041).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d artifacts/checkpoints/best ]]; then
  echo "ERROR: artifacts/checkpoints/best missing. Back up or train first." >&2
  exit 1
fi

python scripts/evaluate_gemma4.py \
  --model E2B \
  --checkpoint artifacts/checkpoints/best \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-eval-run-chuncked \
  --chunk_length_s 30 \
  --batch-size 4 \
  --normalize jiwer_default

echo "Wrote eval/gemma4-eval-run-chuncked/metrics.json"
python - <<'PY'
import json
from pathlib import Path

ref = json.loads(Path("configs/baseline/metrics_reference.json").read_text())
got = json.loads(Path("eval/gemma4-eval-run-chuncked/metrics.json").read_text())
wn_ref = ref["pooled"]["wer_normalized"]
wn_got = got["pooled"]["wer_normalized"]
print(f"wer_normalized: {wn_got:.6f}  (reference {wn_ref:.6f}, delta {wn_got - wn_ref:+.6f})")
PY
