# src/utils/paths.py — artifact directories under the project root.
from __future__ import annotations

from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = WORK_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

PREPARED_LOCAL = ARTIFACTS_DIR / "prepared_dataset"
RETENTION_PREPARED_LOCAL = ARTIFACTS_DIR / "retention_prepared_dataset"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
BASELINE_JSON = ARTIFACTS_DIR / "baseline_results.json"
FINETUNED_JSON = ARTIFACTS_DIR / "finetuned_results.json"
RETENTION_BASELINE_JSON = ARTIFACTS_DIR / "retention_baseline_results.json"
RETENTION_FINETUNED_JSON = ARTIFACTS_DIR / "retention_finetuned_results.json"
MERGED_LOCAL = ARTIFACTS_DIR / "merged"
