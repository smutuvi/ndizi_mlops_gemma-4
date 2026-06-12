"""Shared Gemma 4 PEFT loader for inference scripts (KV-shared-layer safe)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.gemma4_lora import load_gemma4_peft_adapter  # noqa: E402

__all__ = ["load_gemma4_peft_adapter"]
