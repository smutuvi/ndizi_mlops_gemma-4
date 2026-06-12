#!/usr/bin/env python3
"""Rewrite adapter_config.json for Gemma 4 KV-shared layers (fix PEFT missing-key warnings)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForMultimodalLM  # noqa: E402

from src.models.gemma4_lora import rewrite_adapter_config_for_kv_shared  # noqa: E402
from src.utils.runtime import get_runtime  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "adapter_dir",
        type=Path,
        help="Local adapter folder (must contain adapter_config.json)",
    )
    p.add_argument(
        "--base-model-id",
        default=None,
        help="Base model id (default: runtime base_model_id)",
    )
    args = p.parse_args()

    rt = get_runtime()
    base_id = args.base_model_id or rt.base_model_id
    adapter_dir = args.adapter_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise SystemExit(f"Not a directory: {adapter_dir}")

    print(f"Loading base config from {base_id} (CPU, bf16)...")
    base = AutoModelForMultimodalLM.from_pretrained(
        base_id,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if rewrite_adapter_config_for_kv_shared(adapter_dir, base):
        print(f"Patched {adapter_dir / 'adapter_config.json'}")
    else:
        print("No patch needed (config already KV-shared-safe or not a LoRA adapter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
