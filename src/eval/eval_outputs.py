# src/eval/eval_outputs.py — metrics.json / predictions.json / predictions.csv writers.
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.eval.normalize import extra_normalized_fields_for_row, try_build_jiwer_transforms, utterance_wer_cer


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_predictions_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_prediction_rows(
    *,
    dataset_key: str,
    refs: list[str],
    hyps: list[str],
    groups: list[str | None],
    normalize: str,
    row_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jiwer_tr_w = jiwer_tr_c = None
    if normalize == "jiwer_default":
        jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        g = groups[i] if groups else None
        meta = row_meta[i] if row_meta and i < len(row_meta) else {}
        wer, cer = utterance_wer_cer(ref, hyp, "none")
        rec: dict[str, Any] = {
            "dataset": dataset_key,
            "source_dataset": g,
            "row_idx": meta.get("row_idx", i),
            "reference": ref,
            "prediction": hyp,
            "wer": wer,
            "cer": cer,
            "decode_wall_s": meta.get("decode_wall_s"),
            "rtfx": meta.get("rtfx"),
        }
        if meta.get("audio_path"):
            rec["audio_path"] = meta["audio_path"]
        if meta.get("audio_duration_s") is not None:
            rec["audio_duration_s"] = meta["audio_duration_s"]
        if normalize != "none":
            rec.update(
                extra_normalized_fields_for_row(
                    ref, hyp, normalize, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c
                )
            )
            rec["rtfx_normalized"] = rec.get("rtfx")
        rows.append(rec)
    return rows
