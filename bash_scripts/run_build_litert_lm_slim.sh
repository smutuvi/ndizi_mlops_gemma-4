#!/usr/bin/env bash
# Build smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm-slim (Google E2B shell + Ndizi LLM).
set -euo pipefail
cd "$(dirname "$0")/.."

conda activate ndizi 2>/dev/null || true

python -m pip install -q litert-lm-builder pillow torchvision 2>/dev/null || true

python scripts/build_litert_lm_slim.py "$@"
