# src/eval/hub_datasets.py — load Hub test splits for batch eval (ndizi_mlops-style specs).
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Audio, Dataset, load_dataset

from src.utils.constants import AUDIO_COLUMN, TARGET_SR, TEXT_COLUMN


@dataclass
class SplitSpec:
    dataset_id: str
    split: str

    @classmethod
    def parse(cls, raw: str) -> SplitSpec:
        if ":" not in raw:
            return cls(raw.strip(), "test")
        ds, sp = raw.split(":", 1)
        return cls(ds.strip(), sp.strip())


def resolve_columns(column_names: list[str]) -> tuple[str, str]:
    cols_l = {c.lower(): c for c in column_names}
    audio_candidates = [AUDIO_COLUMN, "audio", "speech", "path"]
    text_candidates = [
        TEXT_COLUMN,
        "text",
        "transcript",
        "sentence",
        "transcription",
        "normalized_text",
        "clean_transcription",
    ]
    audio_col = next((cols_l[c.lower()] for c in audio_candidates if c.lower() in cols_l), None)
    text_col = next((cols_l[c.lower()] for c in text_candidates if c.lower() in cols_l), None)
    if not audio_col or not text_col:
        raise ValueError(
            f"Could not resolve audio/text columns from {column_names}. "
            f"Pass --audio-column and --text-column."
        )
    return audio_col, text_col


def _standardize_columns(ds: Dataset, audio_col: str, text_col: str) -> Dataset:
    if audio_col != AUDIO_COLUMN:
        ds = ds.rename_column(audio_col, AUDIO_COLUMN)
    if text_col != TEXT_COLUMN:
        ds = ds.rename_column(text_col, TEXT_COLUMN)
    return ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SR))


def max_clip_duration_s(ds: Dataset, sample_n: int | None = None) -> float:
    n = len(ds) if sample_n is None else min(sample_n, len(ds))
    m = 0.0
    for i in range(n):
        a = ds[i][AUDIO_COLUMN]
        if a and "array" in a and "sampling_rate" in a:
            m = max(m, len(a["array"]) / float(a["sampling_rate"]))
    return m


def load_hub_eval_splits(
    specs: list[str],
    *,
    max_samples: int | None = None,
    dataset_revision: str | None = None,
    audio_column: str | None = None,
    text_column: str | None = None,
) -> dict[str, Dataset]:
    kw: dict[str, Any] = {}
    if dataset_revision:
        kw["revision"] = dataset_revision.strip()

    out: dict[str, Dataset] = {}
    for raw in specs:
        spec = SplitSpec.parse(raw)
        key = f"{spec.dataset_id}:{spec.split}"
        print(f"[eval] Loading {key}...")
        ds = load_dataset(spec.dataset_id, split=spec.split, **kw)
        if audio_column and text_column:
            a_col, t_col = audio_column, text_column
        else:
            a_col, t_col = resolve_columns(list(ds.column_names))
        ds = _standardize_columns(ds, a_col, t_col)
        ds = ds.add_column("source_dataset", [spec.dataset_id] * len(ds))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        out[key] = ds
        print(f"[eval]   {len(ds):,} rows")
    return out
