# src/eval/finetuned.py — evaluate LoRA-tuned Gemma 4 on domain + retention suites.
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.eval.eval_outputs import (
    build_prediction_rows,
    write_metrics_json,
    write_predictions_csv,
    write_predictions_json,
)
from src.eval.hub_datasets import load_hub_eval_splits, max_clip_duration_s
from src.eval.metrics import eval_with, print_grouped, score_grouped
from src.eval.normalize import (
    TEXT_NORMALIZE_DEFAULT,
    TEXT_NORMALIZE_EVAL_DEFAULT,
    pooled_wer_cer,
    try_build_jiwer_transforms,
    utterance_wer_cer,
)
from src.inference.chunked_transcribe import make_gemma_predict_fn, resolve_chunk_length_s
from src.inference.gemma_inputs import load_audio_file
from src.inference.transcribe import gemma_transcribe
from src.utils.constants import AUDIO_COLUMN, TEXT_COLUMN
from src.utils.paths import (
    CHECKPOINT_DIR,
    FINETUNED_JSON,
    PREDICTIONS_DIR,
    PREPARED_LOCAL,
    RETENTION_FINETUNED_JSON,
    RETENTION_PREPARED_LOCAL,
)
from src.utils.runtime import get_runtime


def load_finetuned_gemma(checkpoint_dir: Path | str | None = None, *, fp16: bool = False):
    """Load base Gemma 4 + LoRA adapter for inference."""
    rt = get_runtime()
    adapter = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR / "best"
    if not adapter.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint not found: {adapter}")

    dtype = torch.float16 if fp16 else torch.bfloat16
    processor = AutoProcessor.from_pretrained(rt.base_model_id, padding_side="left")
    base = AutoModelForMultimodalLM.from_pretrained(
        rt.base_model_id,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    return model, processor, adapter


def _filter_by_max_duration(ds, max_sec: float | None):
    if max_sec is None:
        return ds, 0
    keep = []
    for i in range(len(ds)):
        a = ds[i][AUDIO_COLUMN]
        dur = len(a["array"]) / float(a["sampling_rate"])
        if dur <= max_sec:
            keep.append(i)
    dropped = len(ds) - len(keep)
    return ds.select(keep) if dropped else ds, dropped


def _row_meta(ds) -> list[dict]:
    meta = []
    for i in range(len(ds)):
        a = ds[i][AUDIO_COLUMN]
        dur = None
        if a and "array" in a and "sampling_rate" in a:
            dur = len(a["array"]) / float(a["sampling_rate"])
        meta.append({"row_idx": i, "audio_duration_s": dur})
    return meta


def _apply_aggressive_qc(ds):
    """Optional QC filter (slow). Returns (filtered_ds, dropped, reason_counts)."""
    from collections import Counter

    from src.data.qc import QCConfig, check_example

    cfg = QCConfig()
    keep_idx = []
    reasons = Counter()
    for i in range(len(ds)):
        row = ds[i]
        ok, reason = check_example(row.get(AUDIO_COLUMN), str(row.get(TEXT_COLUMN) or ""), cfg)
        if ok:
            keep_idx.append(i)
        else:
            reasons[reason] += 1
    dropped = len(ds) - len(keep_idx)
    return (ds.select(keep_idx) if dropped else ds), dropped, dict(reasons)


def _load_eval_tests(args) -> dict:
    if getattr(args, "test_datasets", None):
        return load_hub_eval_splits(
            args.test_datasets,
            max_samples=getattr(args, "max_samples", None),
            dataset_revision=getattr(args, "dataset_revision", None),
            audio_column=getattr(args, "audio_column", None),
            text_column=getattr(args, "text_column", None),
        )

    dsd = load_from_disk(str(PREPARED_LOCAL))
    tests = {"domain_test": dsd["test"]}
    if getattr(args, "retention_eval", False) and RETENTION_PREPARED_LOCAL.exists():
        ret_dd = load_from_disk(str(RETENTION_PREPARED_LOCAL))
        if "test" in ret_dd:
            tests["retention_test"] = ret_dd["test"]
    return tests


def run_transcribe_file(args) -> None:
    """Transcribe one audio file with the finetuned LoRA adapter."""
    audio_path = Path(args.audio).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else CHECKPOINT_DIR / "best"
    chunk_s = getattr(args, "chunk_length_s", None)
    stride_s = getattr(args, "stride_length_s", None)

    print(f"[transcribe] audio      = {audio_path}")
    print(f"[transcribe] checkpoint = {checkpoint.resolve()}")
    if chunk_s:
        print(f"[transcribe] chunking   = {chunk_s}s stride={stride_s}")

    model, processor, adapter = load_finetuned_gemma(
        checkpoint, fp16=bool(getattr(args, "fp16", False))
    )
    audio = load_audio_file(audio_path)
    if chunk_s is None:
        from src.inference.gemma_inputs import resample_mono_16k

        dur = len(resample_mono_16k(audio)) / 16000.0
        chunk_s = resolve_chunk_length_s(None, max_clip_duration_s=dur, auto_chunk_long=True)
    max_new = int(getattr(args, "max_new_tokens", 256))
    rep_pen = getattr(args, "repetition_penalty", None)
    nrep = getattr(args, "no_repeat_ngram_size", None)
    predict = make_gemma_predict_fn(
        model,
        processor,
        chunk_length_s=chunk_s,
        stride_length_s=stride_s,
        max_new_tokens=max_new,
        repetition_penalty=rep_pen,
        no_repeat_ngram_size=nrep,
    )
    hyp = predict([audio])[0]

    print("\n--- transcription ---")
    print(hyp)
    print("---------------------")

    result = {
        "audio": str(audio_path),
        "checkpoint": str(adapter.resolve()),
        "hypothesis": hyp,
        "chunk_length_s": chunk_s,
        "stride_length_s": stride_s,
    }
    ref = getattr(args, "reference", None)
    if ref:
        norm = getattr(args, "normalize", TEXT_NORMALIZE_EVAL_DEFAULT)
        wer, cer = utterance_wer_cer(ref, hyp, norm)
        result["reference"] = ref
        result["normalize"] = norm
        result["wer"] = wer
        result["cer"] = cer
        print(f"\nWER={wer:.4f}  CER={cer:.4f}  (normalize={norm})")

    out = getattr(args, "output", None)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Wrote", out_path)


def run_evaluate(args) -> None:
    checkpoint = getattr(args, "checkpoint", None)
    text_mode = getattr(args, "normalize", TEXT_NORMALIZE_EVAL_DEFAULT)
    out_dir = Path(getattr(args, "output_dir", None) or PREDICTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, adapter = load_finetuned_gemma(
        checkpoint, fp16=bool(getattr(args, "fp16", False))
    )
    tests = _load_eval_tests(args)

    chunk_cli = getattr(args, "chunk_length_s", None)
    stride_s = getattr(args, "stride_length_s", None)
    max_audio_seconds = getattr(args, "max_audio_seconds", None)
    auto_chunk = not bool(getattr(args, "no_auto_chunk", False))
    max_new = int(getattr(args, "max_new_tokens", 256))
    rep_pen = getattr(args, "repetition_penalty", None)
    nrep = getattr(args, "no_repeat_ngram_size", None)

    all_refs: list[str] = []
    all_hyps: list[str] = []
    all_groups: list[str | None] = []
    per_set: dict = {}
    predictions_out: list[dict] = []
    results: dict = {"text_normalize": text_mode}

    for name, split in tests.items():
        max_dur = max_clip_duration_s(split)
        chunk_s = resolve_chunk_length_s(
            chunk_cli, max_clip_duration_s=max_dur, auto_chunk_long=auto_chunk
        )
        if chunk_s and chunk_cli is None:
            print(
                f"[finetuned] {name}: max clip {max_dur:.1f}s > {chunk_s:.0f}s — "
                f"auto chunk_length_s={chunk_s}"
            )

        eval_split = split
        dropped = 0
        if max_audio_seconds is not None and chunk_s is None:
            eval_split, dropped = _filter_by_max_duration(split, max_audio_seconds)
            if dropped:
                print(f"[finetuned] {name}: dropped {dropped} clips > {max_audio_seconds}s")

        dropped_qc = 0
        qc_reasons = None
        if bool(getattr(args, "aggressive_qc", False)):
            eval_split, dropped_qc, qc_reasons = _apply_aggressive_qc(eval_split)
            if dropped_qc:
                # Print first few keys only (counts are in metrics.json)
                top = list(qc_reasons.items())[:5]
                print(f"[finetuned] {name}: aggressive_qc dropped {dropped_qc} rows (top={top})")

        print(f"\n[finetuned] {name} ({len(eval_split)} rows, chunk_length_s={chunk_s})")
        predict = make_gemma_predict_fn(
            model,
            processor,
            chunk_length_s=chunk_s,
            stride_length_s=stride_s,
            max_new_tokens=max_new,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=nrep,
        )
        refs, hyps, groups = eval_with(
            predict,
            eval_split,
            batch_size=getattr(args, "batch_size", 4),
            desc=f"finetuned {name}",
        )
        scores_raw = score_grouped(refs, hyps, groups, normalize="none")
        per_set[name] = {
            "wer": scores_raw["pooled"]["wer"],
            "cer": scores_raw["pooled"]["cer"],
            "n": scores_raw["pooled"]["n"],
            "dropped_long": dropped,
            "dropped_qc": dropped_qc,
            "qc_reasons": qc_reasons,
            "chunk_length_s": chunk_s,
        }
        if text_mode != "none":
            scores_norm = score_grouped(refs, hyps, groups, normalize=text_mode)
            per_set[name]["wer_normalized"] = scores_norm["pooled"]["wer"]
            per_set[name]["cer_normalized"] = scores_norm["pooled"]["cer"]
        results[name] = scores_raw

        print_grouped(f"finetuned {name}", scores_raw, normalize="none")

        meta = _row_meta(eval_split)
        predictions_out.extend(
            build_prediction_rows(
                dataset_key=name,
                refs=refs,
                hyps=hyps,
                groups=groups,
                normalize=text_mode,
                row_meta=meta,
            )
        )
        all_refs.extend(refs)
        all_hyps.extend(hyps)
        all_groups.extend(groups)

    pooled_scores = score_grouped(all_refs, all_hyps, all_groups, normalize="none")
    pooled = {
        "wer": pooled_scores.get("pooled", {}).get("wer"),
        "cer": pooled_scores.get("pooled", {}).get("cer"),
        "n_utterances": pooled_scores.get("pooled", {}).get("n"),
    }
    if text_mode != "none":
        jiwer_tr_w = jiwer_tr_c = None
        if text_mode == "jiwer_default":
            jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
        wn, cn = pooled_wer_cer(all_hyps, all_refs, text_mode, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c)
        pooled["wer_normalized"] = wn
        pooled["cer_normalized"] = cn
    metrics_payload = {
        "text_normalize": text_mode,
        "pooled": pooled,
        "per_set": per_set,
        "splits": results,
        "run_info": {
            "checkpoint": str(adapter.resolve()),
            "base_model_id": get_runtime().base_model_id,
            "test_datasets": getattr(args, "test_datasets", None),
            "output_dir": str(out_dir.resolve()),
            "batch_size": getattr(args, "batch_size", 4),
            "chunk_length_s": chunk_cli,
            "stride_length_s": stride_s,
            "max_audio_seconds": max_audio_seconds,
            "fp16": bool(getattr(args, "fp16", False)),
            "max_samples": getattr(args, "max_samples", None),
        },
    }

    write_metrics_json(out_dir / "metrics.json", metrics_payload)
    write_predictions_json(out_dir / "predictions.json", predictions_out)
    write_predictions_csv(out_dir / "predictions.csv", predictions_out)

    FINETUNED_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote", out_dir / "metrics.json")
    print("Wrote", out_dir / "predictions.json")
    print("Wrote", out_dir / "predictions.csv")
    print("Wrote", FINETUNED_JSON)

    if "retention_test" in tests:
        retention_results = {"retention_test": results.get("retention_test")}
        RETENTION_FINETUNED_JSON.write_text(json.dumps(retention_results, indent=2), encoding="utf-8")
        print("Wrote", RETENTION_FINETUNED_JSON)
