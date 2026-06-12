#!/usr/bin/env python3
"""Build a slim LiteRT-LM bundle: Google E2B base + Ndizi finetuned LLM (prefill_decode).

Default Hub target: smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm-slim

Requires on PATH (conda ndizi):
  pip install litert-lm-builder litert-torch-nightly peft pillow torchvision
  # provides: litert-lm-peek, litert-lm-builder, python -m litert_torch.generative.export_hf

─── Full pipeline from LoRA adapter (recommended) ────────────────────────────
  conda activate ndizi
  cd ndizi_mlops_gemma-4
  python scripts/build_litert_lm_slim.py --merge --hf-token YOUR_TOKEN --upload

  Runs: merge LoRA → base model → INT4 export → splice community shell → upload.

─── Skip merge (already merged locally or on HF) ────────────────────────────
  python scripts/build_litert_lm_slim.py \\
    --merged-model /path/to/merged_model \\
    --upload

─── Skip merge + export (reuse an existing ~5 GB .litertlm) ─────────────────
  python scripts/build_litert_lm_slim.py \\
    --skip-export \\
    --finetuned-litertlm /path/to/gemma-4-e2b-sw-asr-ndizi.litertlm \\
    --upload

  Note: size will match the existing bundle (~4-5 GB); use this to fix ASR
  without re-exporting. For the slim ~2.6 GB target, omit --skip-export.

─── Low-RAM official shell (phone OOM on >3 GB; stock ASR, not finetuned) ───
  python scripts/build_litert_lm_slim.py --official-shell --upload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.litert.splice_slim import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_BASE_MODEL,
    DEFAULT_HUB_REPO,
    DEFAULT_MERGED_MODEL,
    DEFAULT_OUTPUT_NAME,
    run_build,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Merge flags (new) ──────────────────────────────────────────────────────
    merge = p.add_argument_group("Merge (LoRA → base model)")
    merge.add_argument(
        "--merge",
        action="store_true",
        help="Merge the LoRA adapter into the base model before exporting. "
             "Outputs to <work-dir>/merged_model and uses it as --merged-model.",
    )
    merge.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help=f"HF adapter repo or local path (default: {DEFAULT_ADAPTER})",
    )
    merge.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"HF base model id or local path (default: {DEFAULT_BASE_MODEL})",
    )
    merge.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token — required for gated models (Gemma) when using --merge",
    )

    # ── Export / splice flags ──────────────────────────────────────────────────
    p.add_argument("--merged-model", default=DEFAULT_MERGED_MODEL,
                   help="HF merged weights id or local path (ignored when --merge is set)")
    p.add_argument("--work-dir", type=Path, default=None, help="Default: artifacts/litert_slim")
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output .litertlm filename")
    p.add_argument("--hub-repo", default=DEFAULT_HUB_REPO, help="HF repo for --upload")
    p.add_argument("--quantization", default="dynamic_wi4_afp32",
                   help="Export quantization recipe for LLM slice (default: dynamic_wi4_afp32 = INT4)")
    p.add_argument("--cache-length", type=int, default=1024,
                   help="KV-cache length for export (default: 1024 — keeps TFLite under 2 GB FlatBuffers limit)")
    p.add_argument("--prefill-lengths", default="[64]",
                   help="Prefill sequence lengths for export (default: [64])")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip litert_torch export; splice from --finetuned-litertlm (e.g. existing 5 GB bundle). "
             "Fixes ASR but does NOT reduce size — omit for the slim ~2.6 GB target.",
    )
    p.add_argument(
        "--finetuned-litertlm",
        type=Path,
        default=None,
        help="Path to finetuned .litertlm (required with --skip-export)",
    )
    p.add_argument("--upload", action="store_true", help="Upload bundle + README to --hub-repo")
    p.add_argument(
        "--official-shell",
        action="store_true",
        help="Copy litert-community E2B ~2.6 GB bundle (low RAM; stock ASR, not Ndizi-finetuned)",
    )

    args = p.parse_args()
    run_build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
