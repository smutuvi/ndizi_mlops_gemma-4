# src/eval/baseline.py — zero-shot Gemma 4 (+ optional Whisper) baseline WER/CER.
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.eval.eval_outputs import (
    build_prediction_rows,
    write_metrics_json,
    write_predictions_csv,
    write_predictions_json,
)
from src.eval.hub_datasets import load_hub_eval_splits, max_clip_duration_s
from src.eval.metrics import eval_with, print_grouped, score_grouped
from src.eval.normalize import TEXT_NORMALIZE_DEFAULT, pooled_wer_cer, try_build_jiwer_transforms
from src.inference.chunked_transcribe import make_gemma_predict_fn, resolve_chunk_length_s
from src.inference.transcribe import gemma_transcribe
from src.utils.constants import TARGET_SR, WHISPER_REF_ID
from src.utils.paths import (
    BASELINE_JSON,
    PREDICTIONS_DIR,
    PREPARED_LOCAL,
    RETENTION_BASELINE_JSON,
    RETENTION_PREPARED_LOCAL,
)
from src.utils.runtime import get_runtime


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

    if getattr(args, "retention_eval", None):
        ret_dd = load_from_disk(str(RETENTION_PREPARED_LOCAL))
        if "test" not in ret_dd:
            raise SystemExit("Retention suite prepared dataset has no 'test' split.")
        tests["retention_test"] = ret_dd["test"]
    return tests


def _row_meta(ds) -> list[dict]:
    meta = []
    for i in range(len(ds)):
        a = ds[i]["audio"]
        dur = None
        if a and "array" in a and "sampling_rate" in a:
            dur = len(a["array"]) / float(a["sampling_rate"])
        meta.append({"row_idx": i, "audio_duration_s": dur})
    return meta


def run_baseline(args) -> None:
    rt = get_runtime()
    normalize = getattr(args, "normalize", TEXT_NORMALIZE_DEFAULT)
    out_dir = Path(getattr(args, "output_dir", None) or PREDICTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = _load_eval_tests(args)

    processor = AutoProcessor.from_pretrained(rt.base_model_id, padding_side="left")
    model = AutoModelForMultimodalLM.from_pretrained(
        rt.base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    chunk_cli = getattr(args, "chunk_length_s", None)
    stride_s = getattr(args, "stride_length_s", None)
    max_audio_seconds = getattr(args, "max_audio_seconds", None)
    auto_chunk = not bool(getattr(args, "no_auto_chunk", False))

    all_refs: list[str] = []
    all_hyps: list[str] = []
    all_groups: list[str | None] = []
    per_set: dict = {}
    predictions_out: list[dict] = []
    results = {"text_normalize": normalize}

    for name, split in tests.items():
        max_dur = max_clip_duration_s(split)
        chunk_s = resolve_chunk_length_s(
            chunk_cli, max_clip_duration_s=max_dur, auto_chunk_long=auto_chunk
        )
        print(f"\n[gemma4 zero-shot] {name} ({len(split)} rows, chunk_length_s={chunk_s})")
        predict = make_gemma_predict_fn(
            model, processor, chunk_length_s=chunk_s, stride_length_s=stride_s
        )
        refs, hyps, groups = eval_with(
            predict,
            split,
            args.batch_size,
            desc=f"gemma4 zero-shot {name}",
        )
        scores_raw = score_grouped(refs, hyps, groups, normalize="none")
        per_set[name] = {
            "wer": scores_raw["pooled"]["wer"],
            "cer": scores_raw["pooled"]["cer"],
            "n": scores_raw["pooled"]["n"],
            "chunk_length_s": chunk_s,
        }
        if normalize != "none":
            scores_norm = score_grouped(refs, hyps, groups, normalize=normalize)
            per_set[name]["wer_normalized"] = scores_norm["pooled"]["wer"]
            per_set[name]["cer_normalized"] = scores_norm["pooled"]["cer"]
        results[f"gemma4_{name}"] = scores_raw
        print_grouped(f"gemma4 zero-shot {name}", scores_raw, normalize="none")

        predictions_out.extend(
            build_prediction_rows(
                dataset_key=f"gemma4_{name}",
                refs=refs,
                hyps=hyps,
                groups=groups,
                normalize=normalize,
                row_meta=_row_meta(split),
            )
        )
        all_refs.extend(refs)
        all_hyps.extend(hyps)
        all_groups.extend(groups)

    if args.with_whisper:
        from transformers import pipeline

        asr = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_REF_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        def whisper_predict(audios):
            return [
                asr(
                    {"array": a["array"], "sampling_rate": TARGET_SR},
                    generate_kwargs={"language": "swahili", "task": "transcribe"},
                )["text"]
                for a in audios
            ]

        for name, split in tests.items():
            print(f"\n[whisper] {name}")
            refs, hyps, groups = eval_with(whisper_predict, split, args.batch_size, desc=f"whisper {name}")
            scores = score_grouped(refs, hyps, groups, normalize=normalize)
            results[f"whisper_{name}"] = scores
            print_grouped(f"whisper {name}", scores, normalize=normalize)

    BASELINE_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote", BASELINE_JSON)

    pooled_scores = score_grouped(all_refs, all_hyps, all_groups, normalize="none")
    pooled = {
        "wer": pooled_scores.get("pooled", {}).get("wer"),
        "cer": pooled_scores.get("pooled", {}).get("cer"),
        "n_utterances": pooled_scores.get("pooled", {}).get("n"),
    }
    if normalize != "none":
        jiwer_tr_w = jiwer_tr_c = None
        if normalize == "jiwer_default":
            jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
        wn, cn = pooled_wer_cer(all_hyps, all_refs, normalize, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c)
        pooled["wer_normalized"] = wn
        pooled["cer_normalized"] = cn

    metrics_payload = {
        "text_normalize": normalize,
        "pooled": pooled,
        "per_set": per_set,
        "run_info": {
            "base_model_id": rt.base_model_id,
            "output_dir": str(out_dir.resolve()),
            "batch_size": getattr(args, "batch_size", 4),
            "chunk_length_s": chunk_cli,
            "stride_length_s": stride_s,
            "max_audio_seconds": max_audio_seconds,
            "max_samples": getattr(args, "max_samples", None),
        },
    }
    write_metrics_json(out_dir / "metrics.json", metrics_payload)
    write_predictions_json(out_dir / "predictions.json", predictions_out)
    write_predictions_csv(out_dir / "predictions.csv", predictions_out)
    print("Wrote", out_dir / "metrics.json")
    print("Wrote", out_dir / "predictions.json")
    print("Wrote", out_dir / "predictions.csv")

    if "retention_test" in tests:
        retention_results = {"gemma4_retention_test": results.get("gemma4_retention_test")}
        RETENTION_BASELINE_JSON.write_text(json.dumps(retention_results, indent=2), encoding="utf-8")
        print("Wrote", RETENTION_BASELINE_JSON)
