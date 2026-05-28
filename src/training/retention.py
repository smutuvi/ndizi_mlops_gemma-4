# src/training/retention.py — optional replay mixing with general Swahili retention data.
from __future__ import annotations

from datasets import load_from_disk

from src.data.splits import split_spec_list
from src.utils.paths import RETENTION_PREPARED_LOCAL


def maybe_load_retention_replay_train(args):
    replay_ratio = float(getattr(args, "replay_ratio", 0.0) or 0.0)
    if replay_ratio <= 0:
        return None, 0.0
    specs = split_spec_list(getattr(args, "retention_datasets", None), default_split="train")
    if not specs:
        return None, 0.0
    if not RETENTION_PREPARED_LOCAL.exists():
        raise SystemExit(
            "Retention suite not prepared. Run prepare with --retention-datasets "
            "(or set replay_ratio=0 to disable)."
        )
    ret = load_from_disk(str(RETENTION_PREPARED_LOCAL))
    if "train" not in ret:
        raise SystemExit("Retention suite has no 'train' split.")
    return ret["train"], replay_ratio
