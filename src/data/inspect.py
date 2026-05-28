# src/data/inspect.py — schema, duration, and speaker-overlap audit.
from __future__ import annotations

from datasets import load_dataset
from tqdm.auto import tqdm

from src.utils.constants import (
    AUDIO_COLUMN,
    MAX_AUDIO_SEC,
    SPEAKER_COLUMN,
    SRC_DATASETS,
    TEXT_COLUMN,
)
from src.utils.paths import ARTIFACTS_DIR


def run_inspect() -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speaker_sets: dict[str, set] = {}
    for name in SRC_DATASETS:
        print(f"\n========== {name} ==========")
        ds = load_dataset(name)
        print(ds)
        first = next(iter(ds.values()))
        print("columns:", first.column_names)

        durations, speakers = [], []
        for split_name, split in ds.items():
            for row in tqdm(split, total=len(split), desc=f"{name}/{split_name}", unit="row", dynamic_ncols=True):
                a = row.get(AUDIO_COLUMN)
                if a and "array" in a and "sampling_rate" in a:
                    durations.append(len(a["array"]) / a["sampling_rate"])
                sp = row.get(SPEAKER_COLUMN)
                if sp is not None:
                    speakers.append(sp)
        speaker_sets[name] = set(speakers)

        if durations:
            s = pd.Series(durations)
            print("duration_s describe:")
            print(s.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
            print("clips >30s:", int((s > MAX_AUDIO_SEC).sum()))
            png = ARTIFACTS_DIR / f"{name.replace('/', '_')}_durations.png"
            plt.figure()
            s.hist(bins=60)
            plt.axvline(MAX_AUDIO_SEC, color="red", linestyle="--", label=f"{MAX_AUDIO_SEC:.0f}s limit")
            plt.xlabel("duration (s)")
            plt.ylabel("count")
            plt.title(f"{name} durations")
            plt.legend()
            plt.tight_layout()
            plt.savefig(png)
            print("wrote", png)

    if len(SRC_DATASETS) >= 2:
        a, b = SRC_DATASETS[0], SRC_DATASETS[1]
        if speaker_sets.get(a) and speaker_sets.get(b):
            overlap = speaker_sets[a] & speaker_sets[b]
            print(f"\nSpeakers in {a}: {len(speaker_sets[a])}")
            print(f"Speakers in {b}: {len(speaker_sets[b])}")
            print(f"Overlap: {len(overlap)} speakers")
            if overlap:
                print("WARNING: speaker overlap -- build splits by speaker to avoid leakage.")
