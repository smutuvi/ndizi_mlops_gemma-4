#!/usr/bin/env bash
# Batch eval for Gemma 4 LoRA ASR (scripts/evaluate_gemma4.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/.cache/huggingface}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_ndizi}"
export LIBROSA_CACHE_DIR="${LIBROSA_CACHE_DIR:-/tmp/librosa_cache_ndizi}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache_ndizi}"
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${LIBROSA_CACHE_DIR}" "${MPLCONFIGDIR}"

if [[ $# -lt 1 ]]; then
  echo "Pass arguments to scripts/evaluate_gemma4.py (--checkpoint, --output_dir, --test_datasets ...)." >&2
  exit 1
fi

exec python3 "${BUNDLE_ROOT}/scripts/evaluate_gemma4.py" "$@"
