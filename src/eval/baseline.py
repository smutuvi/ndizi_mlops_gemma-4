# src/eval/baseline.py — zero-shot Gemma 4 (+ optional Whisper) baseline WER/CER.
from __future__ import annotations

import json

import torch
from datasets import load_from_disk
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.eval.metrics import eval_with, print_grouped, score_grouped
from src.eval.normalize import TEXT_NORMALIZE_DEFAULT
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


def run_baseline(args) -> None:
    rt = get_runtime()
    normalize = getattr(args, "normalize", TEXT_NORMALIZE_DEFAULT)
    dsd = load_from_disk(str(PREPARED_LOCAL))
    tests = {"domain_test": dsd["test"]}

    if getattr(args, "retention_eval", None):
        ret_dd = load_from_disk(str(RETENTION_PREPARED_LOCAL))
        if "test" not in ret_dd:
            raise SystemExit("Retention suite prepared dataset has no 'test' split.")
        tests["retention_test"] = ret_dd["test"]

    processor = AutoProcessor.from_pretrained(rt.base_model_id, padding_side="left")
    model = AutoModelForMultimodalLM.from_pretrained(
        rt.base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    PREDICTIONS_DIR.mkdir(exist_ok=True, parents=True)
    results = {"normalize": normalize}
    for name, split in tests.items():
        print(f"\n[gemma4 zero-shot] {name} ({len(split)} rows)")
        refs, hyps, groups = eval_with(
            lambda a: gemma_transcribe(model, processor, a),
            split,
            args.batch_size,
            desc=f"gemma4 zero-shot {name}",
        )
        scores = score_grouped(refs, hyps, groups, normalize=normalize)
        results[f"gemma4_{name}"] = scores
        print_grouped(f"gemma4 zero-shot {name}", scores, normalize=normalize)

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

    if "retention_test" in tests:
        retention_results = {"gemma4_retention_test": results.get("gemma4_retention_test")}
        RETENTION_BASELINE_JSON.write_text(json.dumps(retention_results, indent=2), encoding="utf-8")
        print("Wrote", RETENTION_BASELINE_JSON)
