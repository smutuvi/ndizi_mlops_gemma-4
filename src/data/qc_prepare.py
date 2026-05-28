# src/data/qc_prepare.py — apply multi-gate QC when building prepared datasets.
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

from src.data.qc import QCConfig, check_example
from src.eval.normalize import simple_normalize
from src.utils.constants import AUDIO_COLUMN, MAX_AUDIO_SEC, TEXT_COLUMN

logger = logging.getLogger(__name__)


def qc_config_from_prepare_args(args) -> QCConfig:
    """Build QC thresholds from prepare CLI flags (off unless --aggressive-qc)."""
    cfg = QCConfig()
    if bool(getattr(args, "aggressive_qc", False)) and bool(getattr(args, "qc_chunk_long_with_mms_fa", False)):
        chunk_s = float(getattr(args, "qc_chunk_seconds", MAX_AUDIO_SEC))
        cfg.max_dur = max(float(cfg.max_dur), chunk_s + 0.25)
    return cfg


def _text_for_qc(row: dict, *, use_may6: bool) -> str:
    raw = str(row.get(TEXT_COLUMN) or "").strip()
    if use_may6:
        return simple_normalize(raw)
    return raw


def apply_qc_filter_dataset(
    dataset: Dataset,
    cfg: QCConfig,
    *,
    split_label: str = "",
    audio_only: bool = False,
    use_may6_text: bool = False,
) -> tuple[Dataset, int, dict[str, int]]:
    """Filter HF Dataset rows; return (filtered, dropped_count, reason_counts)."""
    n_before = len(dataset)
    counters: Counter[str] = Counter()

    def _keep(ex: dict) -> bool:
        audio = ex[AUDIO_COLUMN]
        text = _text_for_qc(ex, use_may6=use_may6_text)
        keep, reason = check_example(audio, text, cfg)
        counters[reason] += 1
        return keep

    filtered = dataset.filter(_keep, desc=f"qc {split_label}".strip())
    n_after = len(filtered)
    dropped = n_before - n_after
    label = f"[{split_label}] " if split_label else ""
    logger.info(
        "%sQC: %d → %d rows (dropped %d / %.1f%%)",
        label,
        n_before,
        n_after,
        dropped,
        100.0 * dropped / max(n_before, 1),
    )
    for k, v in sorted(counters.items(), key=lambda kv: (-kv[1], kv[0])):
        if k == "ok" and v == n_after:
            continue
        logger.info("  %-14s  %6d  (%.1f%%)", k, v, 100.0 * v / max(n_before, 1))
    return filtered, dropped, dict(counters)


def prepare_split_with_qc(
    split: Dataset,
    cfg: QCConfig,
    *,
    split_name: str,
    use_may6_text: bool = False,
) -> tuple[Dataset, dict[str, Any]]:
    """QC one split; returns (dataset, stats dict)."""
    out, dropped, reasons = apply_qc_filter_dataset(
        split, cfg, split_label=split_name, audio_only=False, use_may6_text=use_may6_text
    )
    return out, {"dropped": dropped, "n_before": len(split) + dropped, "n_after": len(out), "reasons": reasons}
