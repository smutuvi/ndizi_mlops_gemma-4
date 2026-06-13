# src/inference/transcribe.py — Gemma 4 ASR decode (batch size 1 for audio).
from __future__ import annotations

import torch

from src.eval.normalize import polish_transcript_casing
from src.inference.gemma_inputs import gemma_build_inputs, gemma_inputs_to_device, prepare_audio_for_gemma
from src.utils.constants import ASR_INSTRUCTION


def gemma_transcribe(
    model,
    processor,
    audios,
    instruction: str = ASR_INSTRUCTION,
    *,
    casing_polish: bool = True,
):
    results = []
    for a in audios:
        wave = prepare_audio_for_gemma(a)
        inputs = gemma_build_inputs(processor, wave, instruction, add_generation_prompt=True)
        inputs = gemma_inputs_to_device(inputs, model)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, min_new_tokens=4, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[-1] :]
        hyp = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        if casing_polish:
            hyp = polish_transcript_casing(hyp)
        results.append(hyp)
    return results
