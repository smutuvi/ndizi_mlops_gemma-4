# src/data/leakage.py — block transcripts already used by Sunflower / Gemma 4.
from __future__ import annotations

import re
import unicodedata

from datasets import Dataset, load_dataset

from src.eval.hub_datasets import resolve_columns

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_transcript(text: str | None) -> str:
    """Lowercase, strip punctuation/diacritics — used for anti-join, not for WER."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.casefold()
    t = _PUNCT_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def load_blocked_transcripts(*, include_fleurs: bool = True, include_salt: bool = True) -> set[str]:
    """Collect normalized transcripts Sunflower / Gemma 4 already saw.

    Failures are warnings (gated Hub repos) — never silently train on FLEURS/SALT
    if the load succeeded.
    """
    blocked: set[str] = set()

    if include_fleurs:
        try:
            for split in ("train", "validation", "test"):
                ds = load_dataset("google/fleurs", "sw_ke", split=split)
                col = "transcription" if "transcription" in ds.column_names else None
                if col is None:
                    _, col = resolve_columns(list(ds.column_names))
                for row in ds:
                    blocked.add(normalize_transcript(row.get(col)))
            print(f"[leakage] FLEURS sw_ke: {len(blocked)} blocked transcripts so far")
        except Exception as exc:
            print(f"[leakage-warn] Could not load google/fleurs sw_ke ({exc}). FLEURS overlap will not be filtered.")

    if include_salt:
        n_before = len(blocked)
        try:
            ds = load_dataset("Sunbird/salt", "text-all")
            split = ds["train"] if "train" in ds else ds[next(iter(ds))]
            col = "swa_text" if "swa_text" in split.column_names else None
            if col:
                for row in split:
                    blocked.add(normalize_transcript(row.get(col)))
                print(f"[leakage] Sunbird/salt swa_text: +{len(blocked) - n_before} transcripts")
        except Exception as extra:
            print(f"[leakage-warn] Could not load Sunbird/salt ({extra}). SALT overlap will not be filtered.")

    blocked.discard("")
    print(f"[leakage] Total blocked transcript hashes: {len(blocked)}")
    return blocked


def filter_blocked_rows(ds: Dataset, text_column: str, blocked: set[str]) -> Dataset:
    if not blocked:
        return ds
    keep = [
        i for i, row in enumerate(ds)
        if normalize_transcript(row.get(text_column)) not in blocked
    ]
    dropped = len(ds) - len(keep)
    if dropped:
        print(f"[leakage] Dropped {dropped:,} / {len(ds):,} rows matching blocked transcripts")
    return ds.select(keep) if dropped else ds


def parse_asr_repo(raw: str) -> tuple[str, str | None]:
    """Parse ``org/name`` or ``org/name:config`` Hub ids."""
    raw = raw.strip()
    if ":" in raw:
        repo, cfg = raw.rsplit(":", 1)
        if "/" in repo and cfg:
            return repo, cfg
    return raw, None


def load_asr_hub_dataset(repo_id: str, *, config: str | None = None):
    """Load a Hub ASR dataset and rename audio/text columns to the project contract."""
    from datasets import Audio, DatasetDict

    from src.utils.constants import AUDIO_COLUMN, TARGET_SR, TEXT_COLUMN

    ds = load_dataset(repo_id, config) if config else load_dataset(repo_id)
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})
    out = DatasetDict()
    for split, part in ds.items():
        a_col, t_col = resolve_columns(list(part.column_names))
        if a_col != AUDIO_COLUMN:
            part = part.rename_column(a_col, AUDIO_COLUMN)
        if t_col != TEXT_COLUMN:
            part = part.rename_column(t_col, TEXT_COLUMN)
        part = part.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SR))
        out[split] = part
    return out
