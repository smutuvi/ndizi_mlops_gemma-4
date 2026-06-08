# additional_scripts/eval_io.py — metrics.json / predictions.json / predictions.csv (ndizi_mlops shape).
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import jiwer

# Shared with src/utils/constants.py — keep in sync for training + Colab inference scripts.
DEFAULT_ASR_INSTRUCTION = (
    "Transcribe the following speech segment in Swahili into Swahili text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* Use natural written Swahili capitalization: uppercase at the start of each sentence; "
    "uppercase for proper nouns and spoken labels (e.g. Aina A, Aina B) when the speaker uses them.\n"
    "* Do not write the whole transcript in lowercase; preserve uppercase and lowercase as in normal Swahili writing.\n"
    "* Use standard Swahili punctuation (periods, commas, question marks) that matches the speech.\n"
    "* Do not repeat the same word or phrase; transcribe each word once.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
    "and write 3 instead of three."
)

# Stronger punctuation variant for inference — use when the model under-punctuates.
# Safe to use at inference time even if training used DEFAULT_ASR_INSTRUCTION.
PUNCTUATION_ASR_INSTRUCTION = (
    "Transcribe the following speech segment in Swahili into Swahili text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* Use natural written Swahili capitalization: uppercase at the start of each sentence; "
    "uppercase for proper nouns and spoken labels (e.g. Aina A, Aina B) when the speaker uses them.\n"
    "* Do not write the whole transcript in lowercase; preserve uppercase and lowercase as in normal Swahili writing.\n"
    "* PUNCTUATION IS MANDATORY — a transcription with no punctuation is wrong.\n"
    "* End every declarative sentence with a period (.).\n"
    "* End every question with a question mark (?).\n"
    "* Use commas (,) to separate listed items, after introductory phrases (e.g. 'Kwa mfano,'), "
    "and at natural spoken pauses within a long sentence.\n"
    "* Example of correct punctuation: 'Aina ya kwanza ni A, aina ya pili ni B. Je, unaelewa?'\n"
    "* Example of wrong punctuation: 'Aina ya kwanza ni A aina ya pili ni B Je unaelewa'\n"
    "* Do not repeat the same word or phrase; transcribe each word once.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
    "and write 3 instead of three."
)


# Short variant for training examples in asr_safe / asr_moderate modes.
# Reduces instruction-following contamination of the LM decoder.
SHORT_ASR_INSTRUCTION = (
    "Transcribe the Swahili audio exactly as spoken. "
    "Output only the transcript text, no explanations."
)


def try_build_jiwer_transforms() -> tuple[Any, Any]:
    tr_w = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    tr_c = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(),
        jiwer.ReduceToListOfListOfChars(),
    ])
    return tr_w, tr_c


def pooled_wer_cer(
    hyps: list[str],
    refs: list[str],
    mode: str = "none",
    *,
    jiwer_tr_w: Any = None,
    jiwer_tr_c: Any = None,
) -> tuple[float, float]:
    pairs = [(h, r) for h, r in zip(hyps, refs) if str(r).strip()]
    if not pairs:
        raise ValueError("No non-empty references to score")
    pl, rl = [h for h, _ in pairs], [r for _, r in pairs]
    if mode == "jiwer_default":
        if jiwer_tr_w is None or jiwer_tr_c is None:
            jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
        wer = jiwer.wer(rl, pl, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w)
        cer = jiwer.cer(rl, pl, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c)
    else:
        wer = jiwer.wer(rl, pl)
        cer = jiwer.cer(rl, pl)
    return float(wer), float(cer)


def utterance_wer_cer(
    ref: str,
    hyp: str,
    mode: str = "none",
    *,
    jiwer_tr_w: Any = None,
    jiwer_tr_c: Any = None,
) -> tuple[float, float]:
    w, c = pooled_wer_cer([hyp], [ref], mode, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c)
    return w, c


def extra_normalized_fields_for_row(
    ref_raw: str,
    pred_raw: str,
    mode: str,
    *,
    jiwer_tr_w: Any = None,
    jiwer_tr_c: Any = None,
) -> dict[str, Any]:
    if mode == "none":
        return {}
    if jiwer_tr_w is None or jiwer_tr_c is None:
        jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
    wo = jiwer.process_words(
        ref_raw, pred_raw, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w
    )
    co = jiwer.process_characters(
        ref_raw, pred_raw, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c
    )
    rn = " ".join(wo.references[0]) if wo.references and wo.references[0] else ""
    pn = " ".join(wo.hypotheses[0]) if wo.hypotheses and wo.hypotheses[0] else ""
    return {
        "text_normalized": rn,
        "prediction_normalized": pn,
        "wer_normalized": float(wo.wer),
        "cer_normalized": float(co.cer),
    }


def _basename_label(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    return os.path.basename(s.replace("\\", "/"))


def resolve_eval_columns(column_names: list[str]) -> tuple[str, str]:
    """Resolve audio and text columns (aligned with ndizi_mlops evaluate_asr_batch)."""
    cols_l = {c.lower(): c for c in column_names}
    audio_candidates = ["audio", "audio_path", "path", "file", "wav", "speech"]
    text_candidates = [
        "text",
        "sentence",
        "transcript",
        "transcription",
        "normalized_text",
        "clean_transcription",
    ]
    audio_col = next((cols_l[c.lower()] for c in audio_candidates if c.lower() in cols_l), None)
    text_col = next((cols_l[c.lower()] for c in text_candidates if c.lower() in cols_l), None)
    if not audio_col or not text_col:
        raise ValueError(
            f"Could not resolve audio/text columns from {column_names}. "
            "Expected Hub ndizi columns (audio + text/sentence/...)."
        )
    return audio_col, text_col


def extract_audio_path_label(
    example: dict[str, Any],
    audio_col: str,
    row_idx: int = 0,
) -> str:
    """
    Stable audio file label for predictions (ndizi_mlops src/data/eval_paths.py).

    Hub parquet often has no audio[\"path\"] after decode; use file_name / case_id / id.
    """
    audio = example.get(audio_col)
    if isinstance(audio, dict):
        for key in ("path", "src", "source", "filename"):
            raw = audio.get(key)
            if raw is not None and str(raw).strip():
                return _basename_label(str(raw))

    for key in (
        "file_name",
        "filename",
        "audio_filename",
        "wav",
        "wav_path",
        "audio_path",
        "path",
        "file",
        "clip",
        "utterance_id",
        "audio_id",
        "case_id",
        "id",
    ):
        if key == audio_col or key not in example:
            continue
        val = example.get(key)
        if val is not None and str(val).strip():
            return _basename_label(str(val))

    return f"row_{row_idx}"


def collect_audio_path_labels(ds, audio_col: str) -> list[str]:
    """Collect labels before Audio(sampling_rate=...) decode (decode=False preserves path when set)."""
    from datasets import Audio

    feat = ds.features.get(audio_col)
    decode_on = getattr(feat, "decode", True) if feat is not None else True
    work = ds.cast_column(audio_col, Audio(decode=False)) if decode_on else ds

    return [extract_audio_path_label(work[i], audio_col, i) for i in range(len(work))]


def build_prediction_rows(
    *,
    dataset: str,
    split: str,
    refs: list[str],
    hyps: list[str],
    row_meta: list[dict[str, Any]] | None = None,
    normalize: str = "none",
    # Legacy kwargs (ignored if dataset/split passed explicitly)
    dataset_key: str | None = None,
    source_dataset: str | None = None,
) -> list[dict[str, Any]]:
    if dataset_key and ":" in dataset_key:
        ds_part, sp_part = dataset_key.split(":", 1)
        dataset = dataset or ds_part
        split = split or sp_part
    if source_dataset and not dataset:
        dataset = source_dataset

    jiwer_tr_w = jiwer_tr_c = None
    if normalize == "jiwer_default":
        jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
    rows: list[dict[str, Any]] = []
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        meta = row_meta[i] if row_meta and i < len(row_meta) else {}
        wer, cer = utterance_wer_cer(ref, hyp, "none")
        rec: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "row_idx": meta.get("row_idx", i),
            "audio_path": meta.get("audio_path", ""),
            "reference": ref,
            "prediction": hyp,
            "wer": wer,
            "cer": cer,
            "decode_wall_s": meta.get("decode_wall_s"),
            "rtfx": meta.get("rtfx"),
        }
        if meta.get("audio_duration_s") is not None:
            rec["audio_duration_s"] = meta["audio_duration_s"]
        if normalize != "none":
            rec.update(
                extra_normalized_fields_for_row(
                    ref, hyp, normalize, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c
                )
            )
        rows.append(rec)
    return rows


def build_metrics_payload(
    *,
    text_normalize: str,
    per_set_scores: dict[str, dict[str, Any]],
    all_refs: list[str],
    all_hyps: list[str],
    chunk_length_s: float,
    run_info: dict[str, Any],
) -> dict[str, Any]:
    """Match evaluate_gemma4.py metrics.json: raw pooled/per_set + optional *_normalized."""
    raw_w, raw_c = pooled_wer_cer(all_hyps, all_refs, "none")
    pooled: dict[str, Any] = {
        "wer": raw_w,
        "cer": raw_c,
        "n_utterances": len(all_refs),
    }
    per_set: dict[str, Any] = {}

    for key, scores in per_set_scores.items():
        entry = {
            "wer": scores["wer"],
            "cer": scores["cer"],
            "n": scores["n"],
            "dropped_long": 0,
            "chunk_length_s": chunk_length_s,
        }
        if text_normalize != "none":
            wn, cn = pooled_wer_cer(scores["hyps"], scores["refs"], text_normalize)
            entry["wer_normalized"] = wn
            entry["cer_normalized"] = cn
        per_set[key] = entry

    if text_normalize != "none":
        wn, cn = pooled_wer_cer(all_hyps, all_refs, text_normalize)
        pooled["wer_normalized"] = wn
        pooled["cer_normalized"] = cn

    return {
        "text_normalize": text_normalize,
        "pooled": pooled,
        "per_set": per_set,
        "run_info": run_info,
    }


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


def write_eval_outputs(
    output_dir: Path,
    *,
    metrics_payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    preds_json_path = output_dir / "predictions.json"
    preds_csv_path = output_dir / "predictions.csv"
    write_metrics_json(metrics_path, metrics_payload)
    write_predictions_json(preds_json_path, prediction_rows)
    write_predictions_csv(preds_csv_path, prediction_rows)
    return metrics_path, preds_json_path, preds_csv_path


def _load_audio_path(path: str) -> dict:
    import numpy as np

    try:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        return {"array": wav.mean(axis=1), "sampling_rate": int(sr)}
    except Exception as exc:
        raise RuntimeError(f"Could not read audio file {path}") from exc


def _load_audio_bytes(data: bytes) -> dict:
    import io

    import numpy as np

    try:
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return {"array": wav.mean(axis=1), "sampling_rate": int(sr)}
    except Exception as exc:
        raise RuntimeError("Could not decode audio bytes") from exc


def as_audio_dict(audio, *, target_sr: int = 16_000) -> dict:
    """
    Normalize HuggingFace audio to {array: float32 mono, sampling_rate: int}.

    Supports classic decoded dicts and datasets 4.x torchcodec AudioDecoder objects.
    """
    import numpy as np

    if audio is None:
        raise ValueError("audio is None")

    if isinstance(audio, dict):
        arr = audio.get("array")
        if arr is not None:
            return {
                "array": np.asarray(arr, dtype=np.float32),
                "sampling_rate": int(audio["sampling_rate"]),
            }
        path = audio.get("path")
        if path:
            return _load_audio_path(path)
        raw = audio.get("bytes")
        if raw is not None:
            return _load_audio_bytes(raw)

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        arr = samples.data
        try:
            import torch

            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
        except ImportError:
            pass
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=-1)
        sr = getattr(samples, "sample_rate", None)
        if sr is None:
            sr = getattr(audio, "sample_rate", target_sr)
        return {"array": np.ascontiguousarray(arr.squeeze()), "sampling_rate": int(sr)}

    raise TypeError(f"Unsupported audio type: {type(audio)!r}")


def polish_transcript_casing(text: str) -> str:
    """
    Light post-decode casing (sentence starts after . ! ?).

    Does not guess proper nouns mid-sentence; the ASR prompt + training labels handle those.
    """
    s = str(text).strip()
    if not s:
        return s
    if s[0].islower():
        s = s[0].upper() + s[1:]
    s = re.sub(
        r"([.!?]\s+)([a-z\u00e0-\u024f])",
        lambda m: m.group(1) + m.group(2).upper(),
        s,
    )
    return s


def collapse_repeated_words(text: str) -> str:
    """Collapse long runs of the same word (decode stutter / loop artifact)."""
    words = str(text).split()
    if len(words) < 2:
        return str(text).strip()
    out: list[str] = []
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        out.append(words[i])
        i = j
    return " ".join(out).strip()


def gemma_generate_kwargs(
    max_new_tokens: int,
    *,
    anti_loop: bool = True,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 4,
) -> dict:
    """Generation kwargs aligned with evaluate_gemma4.py --anti-loop-decode."""
    kw: dict = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if anti_loop:
        kw["repetition_penalty"] = repetition_penalty
        kw["no_repeat_ngram_size"] = no_repeat_ngram_size
    return kw


def audio_duration_s(audio) -> float | None:
    try:
        decoded = as_audio_dict(audio)
    except (TypeError, ValueError, RuntimeError):
        return None
    return len(decoded["array"]) / float(decoded["sampling_rate"])


def eval_row_meta(
    row: dict[str, Any],
    *,
    row_idx: int,
    audio_col: str = "audio",
    audio_path: str | None = None,
) -> dict[str, Any]:
    """Per-utterance metadata for predictions.json (ndizi_mlops-shaped)."""
    label = audio_path if audio_path is not None else extract_audio_path_label(row, audio_col, row_idx)
    meta: dict[str, Any] = {
        "row_idx": row_idx,
        "audio_path": label,
    }
    audio = row.get(audio_col)
    if audio is not None:
        dur = audio_duration_s(audio)
        if dur is not None:
            meta["audio_duration_s"] = dur
    return meta
