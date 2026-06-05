#!/usr/bin/env python3
"""Build a slim LiteRT-LM bundle: Google E2B base + Ndizi finetuned LLM (prefill_decode).

Default Hub target: smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm-slim

Requires on PATH (conda ndizi):
  pip install litert-lm-builder litert-torch-nightly pillow torchvision
  # provides: litert-lm-peek, litert-lm-builder, python -m litert_torch.generative.export_hf

Example (server):
  conda activate ndizi
  cd ndizi_mlops_gemma-4
  python scripts/build_litert_lm_slim.py --upload

Reuse an existing full export (~5GB) without re-running export_hf:
  python scripts/build_litert_lm_slim.py \\
    --skip-export \\
    --finetuned-litertlm /path/to/gemma-4-e2b-sw-asr-ndizi.litertlm \\
    --upload

Phone hangs on ~4GB finetuned rebuild — publish official ~2.6GB shell (not finetuned):
  python scripts/build_litert_lm_slim.py --official-shell --upload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.litert.splice_slim import (  # noqa: E402
    DEFAULT_HUB_REPO,
    DEFAULT_MERGED_MODEL,
    DEFAULT_OUTPUT_NAME,
    run_build,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged-model", default=DEFAULT_MERGED_MODEL, help="HF merged weights id or local path")
    p.add_argument("--work-dir", type=Path, default=None, help="Default: artifacts/litert_slim")
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output .litertlm filename")
    p.add_argument("--hub-repo", default=DEFAULT_HUB_REPO, help="HF repo for --upload")
    p.add_argument("--quantization", default="dynamic_wi4_afp32", help="Export quant recipe for LLM slice")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip litert_torch export; splice from --finetuned-litertlm (e.g. existing 5GB bundle)",
    )
    p.add_argument(
        "--finetuned-litertlm",
        type=Path,
        default=None,
        help="Path to finetuned .litertlm (with --skip-export)",
    )
    p.add_argument("--upload", action="store_true", help="Upload bundle + README to --hub-repo")
    p.add_argument(
        "--official-shell",
        action="store_true",
        help="Copy litert-community E2B ~2.6GB bundle (low RAM; stock ASR, not Ndizi-finetuned)",
    )
    args = p.parse_args()
    run_build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
