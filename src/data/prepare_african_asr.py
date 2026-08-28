# src/data/prepare_african_asr.py — Ndizi + Swahili ASR + WaxalNLP Amharic/Oromo.
from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets

from src.data.leakage import load_asr_hub_split
from src.utils.constants import (
    AFRICAN_ASR_SOURCES,
    AUDIO_COLUMN,
    LANG_ASR_PROMPTS,
    TEXT_COLUMN,
)
from src.utils.paths import AFRICAN_ASR_PREPARED_LOCAL

KEEP = (AUDIO_COLUMN, TEXT_COLUMN, "source_dataset", "task", "language", "asr_instruction")


def _source_label(spec: dict[str, Any]) -> str:
    cfg = spec.get("config")
    return f"{spec['id']}:{cfg}" if cfg else spec["id"]


def _cap(ds: Dataset, n: int | None, *, seed: int = 42) -> Dataset:
    if n is None or n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _drop_empty_text(ds: Dataset) -> Dataset:
    keep = [
        i
        for i, row in enumerate(ds)
        if row.get(TEXT_COLUMN) and str(row[TEXT_COLUMN]).strip()
    ]
    dropped = len(ds) - len(keep)
    if dropped:
        print(f"    dropped {dropped:,} empty-text rows")
        return ds.select(keep)
    return ds


def _tag(ds: Dataset, spec: dict[str, Any]) -> Dataset:
    lang = spec["lang"]
    prompt = LANG_ASR_PROMPTS[lang]
    src = _source_label(spec)
    n = len(ds)
    ds = ds.add_column("source_dataset", [src] * n)
    ds = ds.add_column("task", ["asr"] * n)
    ds = ds.add_column("language", [lang] * n)
    ds = ds.add_column("asr_instruction", [prompt] * n)
    drop = [c for c in ds.column_names if c not in KEEP]
    return ds.remove_columns(drop) if drop else ds


def _load_split(spec: dict[str, Any], split: str) -> Dataset | None:
    try:
        ds = load_asr_hub_split(spec["id"], split, config=spec.get("config"))
    except Exception as exc:
        print(f"  [skip] {_source_label(spec)} {split}: {exc}")
        return None
    ds = _drop_empty_text(ds)
    print(f"  {_source_label(spec):<42} {split:<12} {len(ds):>7,} rows")
    return _tag(ds, spec)


def run_prepare_african_asr(args) -> DatasetDict:
    """Build a multilingual train/val set. Never includes Hub test splits."""
    waxal_max = getattr(args, "waxal_max", None)
    full_waxal = bool(getattr(args, "full_waxal", False))
    sw_p = float(getattr(args, "sw_prob", 0.5))
    am_p = float(getattr(args, "am_prob", 0.25))
    om_p = float(getattr(args, "om_prob", 0.25))

    sources = [dict(s) for s in AFRICAN_ASR_SOURCES]
    extra = getattr(args, "extra_asr", None) or []
    for raw in extra:
        # repo[:config][:lang]  lang defaults to sw
        parts = str(raw).split(":")
        if len(parts) == 1:
            sources.append({"id": parts[0], "config": None, "lang": "sw", "max_train": None})
        elif len(parts) == 2:
            sources.append({"id": parts[0], "config": parts[1], "lang": "sw", "max_train": None})
        else:
            sources.append(
                {"id": parts[0], "config": parts[1] or None, "lang": parts[2], "max_train": None}
            )

    by_lang_train: dict[str, list[Dataset]] = defaultdict(list)
    by_lang_val: dict[str, list[Dataset]] = defaultdict(list)

    for spec in sources:
        cap = spec.get("max_train")
        if spec["id"] == "google/WaxalNLP":
            if full_waxal:
                cap = None
            elif waxal_max is not None:
                cap = int(waxal_max)

        train = _load_split(spec, "train")
        if train is not None:
            train = _cap(train, cap)
            if cap:
                print(f"    capped train → {len(train):,}")
            by_lang_train[spec["lang"]].append(train)

        val = _load_split(spec, "validation")
        if val is not None:
            by_lang_val[spec["lang"]].append(_cap(val, 256))

    if not by_lang_train:
        raise SystemExit("No train data loaded. Check HF_TOKEN and Hub dataset access.")

    lang_trains: dict[str, Dataset] = {}
    for lang, parts in by_lang_train.items():
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        lang_trains[lang] = ds
        print(f"[mix] {lang} train: {len(ds):,} rows")

    order = [lang for lang in ("sw", "am", "om") if lang in lang_trains]
    extra_langs = [lang for lang in lang_trains if lang not in order]
    order.extend(extra_langs)
    weight = {"sw": sw_p, "am": am_p, "om": om_p}
    probs = [max(weight.get(lang, 0.1), 1e-6) for lang in order]
    z = sum(probs)
    probs = [p / z for p in probs]
    print(f"[mix] interleave langs={order} probs={[round(p, 3) for p in probs]}")
    if len(order) == 1:
        train = lang_trains[order[0]]
    else:
        train = interleave_datasets(
            [lang_trains[lang] for lang in order],
            probabilities=probs,
            seed=42,
            stopping_strategy="all_exhausted",
        )

    val_parts = by_lang_val.get("sw") or []
    if not val_parts:
        n = max(1, int(0.02 * len(train)))
        split = train.train_test_split(test_size=n, seed=42)
        train, val = split["train"], split["test"]
        print(f"  (no Swahili validation; held out {n:,} rows from interleaved train)")
    else:
        val = concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
        print(f"[mix] Swahili validation: {len(val):,} rows (checkpoint WER uses this)")

    out = DatasetDict({"train": train, "validation": val})
    print("\nPrepared African ASR mix:")
    for k, v in out.items():
        print(f"  {k}: {len(v):,} rows")
        if "source_dataset" in v.column_names:
            counts = Counter(v["source_dataset"])
            for src in sorted(counts):
                print(f"      {src}: {counts[src]:,}")
        if "language" in v.column_names:
            counts = Counter(v["language"])
            for lang in sorted(counts):
                print(f"      lang={lang}: {counts[lang]:,}")

    dest = Path(getattr(args, "prepared_dir", None) or AFRICAN_ASR_PREPARED_LOCAL)
    if dest.exists():
        shutil.rmtree(dest)
    out.save_to_disk(str(dest))
    print("Saved to", dest)
    return out
