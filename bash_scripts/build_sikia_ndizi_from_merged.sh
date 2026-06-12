#!/usr/bin/env bash
# Build on-device Swahili bundle from smutuvi/gemma-4-e2b-sw-asr-ndizi-merged
# for the Sikia mobile app (LiteRT-LM 0.12).
#
# Requires: conda env with litert-lm-builder, litert-torch-nightly, huggingface_hub
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash bash_scripts/build_sikia_ndizi_from_merged.sh
#
# Uploads to smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm-slim-1 by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python scripts/build_litert_lm_slim.py \
  --merged-model smutuvi/gemma-4-e2b-sw-asr-ndizi-merged \
  --hub-repo smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm-slim-1 \
  --output-name gemma-4-e2b-sw-asr-ndizi-slim.litertlm \
  --upload

echo "Done. Update gemmaE2BNdiziSlimBytes in Sikia model_manager.dart if file size changed."
