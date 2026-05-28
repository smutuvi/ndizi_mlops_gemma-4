# src/data/prepare.py — standardize, chunk, and merge Ndizi Hub datasets to disk.
from __future__ import annotations

import shutil
from collections import Counter

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

from src.data.dataset_loader import load_asr_dataset_specs
from src.data.mms_fa_chunk import add_chunk_index_zero, align_and_chunk_long_clips
from src.data.splits import split_spec_list
from src.utils.constants import AUDIO_COLUMN, MAX_AUDIO_SEC, PREPARED_REPO, SRC_DATASETS, TEXT_COLUMN, TARGET_SR
from src.utils.paths import PREPARED_LOCAL, RETENTION_PREPARED_LOCAL


def run_prepare(args) -> None:
    if args.chunk_test and not args.chunk_long_audio:
        raise SystemExit("--chunk-test requires --chunk-long-audio.")

    prepared: dict[str, DatasetDict] = {}
    for name in SRC_DATASETS:
        print(f"\nPreparing {name}...")
        ds = load_dataset(name)
        ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SR))

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
            if n_long and not args.chunk_long_audio:
                print(
                    f"      WARNING: {n_long} clip(s) exceed Gemma 4's 30s audio limit. "
                    "Re-run with --chunk-long-audio to split them losslessly."
                )
            if split_name == "test" and n_long and args.chunk_long_audio and not args.chunk_test:
                print(
                    f"      WARNING: {n_long} test clip(s) >{MAX_AUDIO_SEC:.0f}s left unchunked; "
                    "add --chunk-test or they may fail at eval time."
                )

        if args.chunk_long_audio:
            chunked = {}
            for split_name, split in ds.items():
                if split_name in ("train", "validation"):
                    chunked[split_name] = align_and_chunk_long_clips(
                        split, add_reassembly=False, desc=f"MMS-FA chunk {split_name}"
                    )
                elif split_name == "test" and args.chunk_test:
                    chunked[split_name] = align_and_chunk_long_clips(
                        split, add_reassembly=True, desc="MMS-FA chunk test"
                    )
                elif split_name == "test":
                    chunked[split_name] = add_chunk_index_zero(split)
                else:
                    chunked[split_name] = split
            ds = DatasetDict(chunked)

        prepared[name] = ds

    out = DatasetDict()
    for split in ("train", "validation", "test"):
        parts = []
        for n in SRC_DATASETS:
            if split in prepared[n]:
                tagged = prepared[n][split].add_column("source_dataset", [n] * len(prepared[n][split]))
                parts.append(tagged)
        if parts:
            out[split] = concatenate_datasets(parts)
        else:
            print(f"  (no '{split}' split in either source dataset)")

    print("\nFinal prepared dataset:")
    for k, v in out.items():
        n_rows = len(v)
        line = f"  {k}: {n_rows:,} rows"
        if k == "test" and "clip_id" in v.column_names:
            n_clips = len(set(v["clip_id"]))
            line += f"  ({n_clips:,} original clips"
            if n_rows != n_clips:
                line += f"; +{n_rows - n_clips:,} rows from --chunk-test)"
            else:
                line += ")"
        print(line)
        if "source_dataset" in v.column_names:
            counts = Counter(v["source_dataset"])
            clips_by_src: dict[str, set] = {}
            if "clip_id" in v.column_names:
                for row in v:
                    clips_by_src.setdefault(row["source_dataset"], set()).add(row["clip_id"])
            for src in sorted(counts):
                detail = f"      {src}: {counts[src]:,} rows"
                if src in clips_by_src:
                    n_clips = len(clips_by_src[src])
                    detail += f" ({n_clips:,} clips"
                    if counts[src] != n_clips:
                        detail += f"; +{counts[src] - n_clips:,} from chunking"
                    detail += ")"
                print(detail)

    if PREPARED_LOCAL.exists():
        shutil.rmtree(PREPARED_LOCAL)
    out.save_to_disk(str(PREPARED_LOCAL))
    print("Saved to", PREPARED_LOCAL)

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.delete_repo(repo_id=PREPARED_REPO, repo_type="dataset", missing_ok=True)
        except Exception as e:
            print(f"  (could not delete existing repo: {e})")
        api.create_repo(repo_id=PREPARED_REPO, repo_type="dataset", private=True, exist_ok=True)
        out.push_to_hub(PREPARED_REPO, private=True)
        print("Pushed to", PREPARED_REPO)

    retention_specs = split_spec_list(getattr(args, "retention_datasets", None), default_split="train")
    if retention_specs:
        print("\nPreparing retention datasets...")
        ret_dd = load_asr_dataset_specs(
            retention_specs,
            suite_label="retention",
            chunk_long_audio=bool(args.chunk_long_audio),
            chunk_test=bool(getattr(args, "retention_chunk_test", False)),
        )
        if RETENTION_PREPARED_LOCAL.exists():
            shutil.rmtree(RETENTION_PREPARED_LOCAL)
        ret_dd.save_to_disk(str(RETENTION_PREPARED_LOCAL))
        print("Saved retention suite to", RETENTION_PREPARED_LOCAL)
