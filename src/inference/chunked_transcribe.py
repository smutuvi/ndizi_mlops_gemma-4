# src/inference/chunked_transcribe.py — fixed-window chunking for long-audio Gemma decode.
from __future__ import annotations

import numpy as np

from src.eval.normalize import polish_transcript_casing
from src.inference.gemma_inputs import resample_mono_16k
from src.inference.transcribe import gemma_transcribe
from src.utils.constants import ASR_INSTRUCTION, MAX_AUDIO_SEC, TARGET_SR


def chunk_waveform(
    wave: np.ndarray,
    *,
    chunk_length_s: float,
    stride_length_s: float | None = None,
    sample_rate: int = TARGET_SR,
) -> list[np.ndarray]:
    """Split waveform into <=chunk_length_s segments (optional overlap via stride)."""
    chunk_samples = max(1, int(chunk_length_s * sample_rate))
    if len(wave) <= chunk_samples:
        return [wave]

    stride_s = chunk_length_s if stride_length_s is None else float(stride_length_s)
    stride_samples = max(1, int(stride_s * sample_rate))
    chunks: list[np.ndarray] = []
    start = 0
    while start < len(wave):
        end = min(start + chunk_samples, len(wave))
        chunks.append(wave[start:end])
        if end >= len(wave):
            break
        start += stride_samples
    return chunks


def gemma_transcribe_chunked(
    model,
    processor,
    audios,
    *,
    chunk_length_s: float = MAX_AUDIO_SEC,
    stride_length_s: float | None = None,
    instruction: str = ASR_INSTRUCTION,
) -> list[str]:
    """Transcribe each clip; long audio is split into chunks and hypotheses are joined."""
    results: list[str] = []
    for audio in audios:
        wave = resample_mono_16k(audio)
        chunks = chunk_waveform(
            wave,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
        )
        parts: list[str] = []
        for chunk in chunks:
            hyp = gemma_transcribe(
                model, processor, [chunk], instruction=instruction, casing_polish=False
            )[0]
            hyp = str(hyp).strip()
            if hyp:
                parts.append(hyp)
        results.append(polish_transcript_casing(" ".join(parts)))
    return results


def resolve_chunk_length_s(
    chunk_length_s: float | None,
    *,
    max_clip_duration_s: float | None,
    auto_chunk_long: bool = True,
) -> float | None:
    """Auto-enable chunking when clips exceed Gemma's single-pass limit (like Whisper pipeline eval)."""
    if chunk_length_s is not None:
        return float(chunk_length_s) if chunk_length_s > 0 else None
    if not auto_chunk_long:
        return None
    limit = float(MAX_AUDIO_SEC)
    if max_clip_duration_s is not None and max_clip_duration_s > limit:
        return limit
    return None


def make_gemma_predict_fn(
    model,
    processor,
    *,
    chunk_length_s: float | None = None,
    stride_length_s: float | None = None,
    instruction: str = ASR_INSTRUCTION,
):
    if chunk_length_s is not None and chunk_length_s > 0:
        cls = chunk_length_s
        st = stride_length_s
        ins = instruction

        def predict(audios):
            return gemma_transcribe_chunked(
                model, processor, audios, chunk_length_s=cls, stride_length_s=st, instruction=ins
            )

        return predict

    ins = instruction
    return lambda audios: gemma_transcribe(model, processor, audios, instruction=ins)
