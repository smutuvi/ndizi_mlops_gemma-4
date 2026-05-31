# src/data/dataset_loader.py — load and merge Hub ASR datasets for Gemma training.
from __future__ import annotations

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

from src.data.mms_fa_chunk import add_chunk_index_zero, align_and_chunk_long_clips
from src.utils.constants import AUDIO_COLUMN, TARGET_SR


def load_asr_dataset_specs(
    specs: list[tuple[str, str]],
    *,
    suite_label: str,
    chunk_long_audio: bool,
    chunk_test: bool,
) -> DatasetDict:
    """Load Hub datasets, tag suite/source, optionally MMS-FA chunk, merge splits."""
    prepared: dict[str, DatasetDict] = {}
    for did, _ in specs:
        ds = load_dataset(did)
        ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SR))
        prepared[did] = ds

    out = DatasetDict()
    all_splits: set[str] = set()
    for ds in prepared.values():
        all_splits |= set(ds.keys())

    for split in sorted(all_splits):
        parts = []
        for did, _ in specs:
            if split not in prepared[did]:
                continue
            part = prepared[did][split].add_column("source_dataset", [did] * len(prepared[did][split]))
            part = part.add_column("suite", [suite_label] * len(part))

            if chunk_long_audio:
                if split in ("train", "validation"):
                    part = align_and_chunk_long_clips(
                        part,
                        add_reassembly=False,
                        desc=f"MMS-FA chunk {suite_label} {split}",
                    )
                elif split == "test" and chunk_test:
                    part = align_and_chunk_long_clips(
                        part,
                        add_reassembly=True,
                        desc=f"MMS-FA chunk {suite_label} test",
                    )
                elif split == "test":
                    part = add_chunk_index_zero(part)
            parts.append(part)
        if parts:
            out[split] = concatenate_datasets(parts)
    return out
