# src/data/prepare_sunflower.py — Ndizi (+ optional ALFFA) with Sunflower leakage filter.
from __future__ import annotations

import shutil
from collections import Counter

from pathlib import Path

from datasets import DatasetDict, concatenate_datasets

from src.data.leakage import (
    filter_blocked_rows,
    load_asr_hub_dataset,
    load_blocked_transcripts,
    parse_asr_repo,
)
from src.data.mms_fa_chunk import add_chunk_index_zero, align_and_chunk_long_clips
from src.utils.constants import AUDIO_COLUMN, MAX_AUDIO_SEC, SUNFLOWER_TRAIN_ASR, TEXT_COLUMN
from src.utils.paths import SUNFLOWER_PREPARED_LOCAL


def _chunk_dd(ds: DatasetDict, *, chunk_long_audio: bool, chunk_test: bool) -> DatasetDict:
    if not chunk_long_audio:
        return ds
    chunked = {}
    for split_name, split in ds.items():
        if split_name in ("train", "validation"):
            chunked[split_name] = align_and_chunk_long_clips(
                split, add_reassembly=False, desc=f"MMS-FA chunk {split_name}"
            )
        elif split_name == "test" and chunk_test:
            chunked[split_name] = align_and_chunk_long_clips(
                split, add_reassembly=True, desc="MMS-FA chunk test"
            )
        elif split_name == "test":
            chunked[split_name] = add_chunk_index_zero(split)
        else:
            chunked[split_name] = split
    return DatasetDict(chunked)


def run_prepare_sunflower(args) -> DatasetDict:
    """Build a leakage-safe train/val/test DatasetDict and save it under artifacts/."""
    train_repos: list[str] = list(getattr(args, "train_datasets", None) or SUNFLOWER_TRAIN_ASR)
    extra = getattr(args, "extra_asr", None)
    if extra:
        extras = extra if isinstance(extra, (list, tuple)) else [extra]
        train_repos.extend(str(x) for x in extras if x)

    blocked: set[str] = set()
    if not getattr(args, "skip_leakage_filter", False):
        blocked = load_blocked_transcripts(
            include_fleurs=not getattr(args, "skip_fleurs_block", False),
            include_salt=not getattr(args, "skip_salt_block", False),
        )

    prepared: dict[str, DatasetDict] = {}
    for name in train_repos:
        print(f"\nPreparing {name}...")
        repo, config = parse_asr_repo(name)
        ds = load_asr_hub_dataset(repo, config=config)
        for split_name, split in ds.items():
            n_total = len(split)
            n_empty = sum(1 for r in split if not r.get(TEXT_COLUMN) or not str(r[TEXT_COLUMN]).strip())
            n_long = sum(
                1
                for r in split
                if len(r[AUDIO_COLUMN]["array"]) / r[AUDIO_COLUMN]["sampling_rate"] > MAX_AUDIO_SEC
            )
            print(
                f"  {split_name:>10}: {n_total:>7,} rows  "
                f"(empty-text: {n_empty}, clips >{MAX_AUDIO_SEC:.0f}s: {n_long})"
            )
        # Never leakage-filter official Ndizi *test* — that is the in-domain eval set.
        is_ndizi = repo.startswith("smutuvi/ndizi")
        filtered = DatasetDict()
        for split_name, split in ds.items():
            if split_name == "test" and is_ndizi:
                filtered[split_name] = split
            else:
                filtered[split_name] = filter_blocked_rows(split, TEXT_COLUMN, blocked)
        ds = _chunk_dd(
            filtered,
            chunk_long_audio=bool(getattr(args, "chunk_long_audio", False)),
            chunk_test=bool(getattr(args, "chunk_test", False)),
        )
        prepared[name] = ds

    out = DatasetDict()
    for split in ("train", "validation", "test"):
        parts = []
        for n in train_repos:
            if split in prepared[n]:
                tagged = prepared[n][split].add_column("source_dataset", [n] * len(prepared[n][split]))
                tagged = tagged.add_column("task", ["asr"] * len(tagged))
                parts.append(tagged)
        if not parts:
            print(f"  (no '{split}' split in source datasets)")
            continue
        common = set(parts[0].column_names)
        for part in parts[1:]:
            common &= set(part.column_names)
        required = {AUDIO_COLUMN, TEXT_COLUMN, "source_dataset", "task"}
        if not required.issubset(common):
            missing = required - common
            raise SystemExit(f"Prepared {split} missing columns {missing}")
        aligned = []
        for part in parts:
            drop = [c for c in part.column_names if c not in common]
            aligned.append(part.remove_columns(drop) if drop else part)
        out[split] = concatenate_datasets(aligned)

    if "train" not in out:
        raise SystemExit("No train split after prepare — check --train-datasets / --extra-asr.")
    if "validation" not in out:
        n = max(1, int(0.02 * len(out["train"])))
        split = out["train"].train_test_split(test_size=n, seed=42)
        out["train"] = split["train"]
        out["validation"] = split["test"]
        print(f"  (no validation split; held out {n:,} rows from train)")

    print("\nFinal leakage-safe dataset:")
    for k, v in out.items():
        print(f"  {k}: {len(v):,} rows")
        if "source_dataset" in v.column_names:
            counts = Counter(v["source_dataset"])
            for src in sorted(counts):
                print(f"      {src}: {counts[src]:,} rows")

    dest = Path(getattr(args, "output_dir", None) or SUNFLOWER_PREPARED_LOCAL)
    if dest.exists():
        shutil.rmtree(dest)
    out.save_to_disk(str(dest))
    print("Saved to", dest)
    return out
