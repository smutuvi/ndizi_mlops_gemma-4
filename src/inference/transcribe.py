# src/inference/transcribe.py — Gemma 4 ASR decode (batch size 1 for audio).
from __future__ import annotations

import torch

from src.inference.gemma_inputs import gemma_build_inputs, gemma_inputs_to_device, prepare_audio_for_gemma
from src.utils.constants import ASR_INSTRUCTION


def gemma_transcribe(
    model,
    processor,
    audios,
    instruction: str = ASR_INSTRUCTION,
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
):
    results = []
    for a in audios:
        wave = prepare_audio_for_gemma(a)
        inputs = gemma_build_inputs(processor, wave, instruction, add_generation_prompt=True)
        inputs = gemma_inputs_to_device(inputs, model)
        gen_kw = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if repetition_penalty is not None:
            gen_kw["repetition_penalty"] = float(repetition_penalty)
        if no_repeat_ngram_size is not None:
            gen_kw["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        new_tokens = out[:, inputs["input_ids"].shape[-1] :]
        results.append(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
    return results
