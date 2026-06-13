# src/inference/gemma_inputs.py — build Gemma 4 multimodal inputs (audio token alignment).
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.constants import ASR_INSTRUCTION, MAX_AUDIO_SEC, TARGET_SR


def load_audio_file(path: str | Path) -> dict:
    """Load a WAV/FLAC/etc. file as a datasets-style audio dict (mono, native sr)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        array = waveform.mean(dim=0).numpy()
        return {"array": array, "sampling_rate": int(sr)}
    except Exception:
        pass

    try:
        import soundfile as sf

        array, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return {"array": array.mean(axis=1), "sampling_rate": int(sr)}
    except Exception as exc:
        raise RuntimeError(f"Could not load audio file {path} (install torchaudio or soundfile)") from exc


def resample_mono_16k(audio) -> np.ndarray:
    """Mono float32 16 kHz waveform (no duration cap)."""
    arr = audio["array"] if isinstance(audio, dict) else audio
    wav = np.asarray(arr, dtype=np.float32)
    if wav.ndim > 1:
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[-1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=-1)
    wav = np.ascontiguousarray(wav.squeeze())
    sr = int(audio.get("sampling_rate", TARGET_SR) if isinstance(audio, dict) else TARGET_SR)
    if sr != TARGET_SR:
        try:
            import torch
            import torchaudio.functional as taf

            wav = taf.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy().astype(np.float32)
        except Exception:
            from scipy.signal import resample_poly

            wav = resample_poly(wav, TARGET_SR, sr).astype(np.float32)
    return wav


def normalize_audio_rms(wav: np.ndarray, target_rms: float = 0.09) -> np.ndarray:
    """Normalize waveform to a target RMS level (matches typical ndizi-1 training energy).

    Quiet clinical recordings (RMS ~0.03) get amplified to training distribution.
    Silent segments (RMS < 1e-6) are returned unchanged to avoid divide-by-zero.
    """
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-6:
        return wav
    return np.clip(wav * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def prepare_audio_for_gemma(audio, *, normalize: bool = True):
    """Mono float32 16 kHz waveform, capped at Gemma's 30 s limit.

    Args:
        normalize: RMS-normalize to training distribution energy (default True).
                   Set False to match original behaviour.
    """
    wav = resample_mono_16k(audio)
    max_samples = int(MAX_AUDIO_SEC * TARGET_SR)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    if normalize:
        wav = normalize_audio_rms(wav)
    return wav


def gemma_soft_token_count(processor, fe_out) -> int:
    mask = fe_out["input_features_mask"][0]
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    num_mel = int(np.asarray(mask, dtype=bool).sum())
    if num_mel <= 0:
        return 0
    t = num_mel
    for _ in range(2):
        t = (t + 2 - 3) // 2 + 1
    return min(t, processor.audio_seq_length)


def gemma_build_inputs(
    processor,
    wave,
    instruction: str = ASR_INSTRUCTION,
    *,
    add_generation_prompt: bool = True,
    assistant_text: str | None = None,
):
    fe_out = processor.feature_extractor([wave], return_tensors="pt")
    n_soft = gemma_soft_token_count(processor, fe_out)
    boa, at, eoa = processor.boa_token, processor.audio_token, processor.eoa_token
    audio_block = f"{boa}{at * n_soft}{eoa}"

    user_content = [
        {"type": "audio", "audio": wave},
        {"type": "text", "text": instruction},
    ]
    messages = [{"role": "user", "content": user_content}]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
        add_generation_prompt = False

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    if isinstance(prompt, list):
        prompt = prompt[0]
    if at not in prompt:
        raise ValueError("Gemma chat template missing audio placeholder token")
    prompt = prompt.replace(at, audio_block, 1)

    inputs = processor(text=prompt, return_tensors="pt", return_mm_token_type_ids=True)
    inputs["input_features"] = fe_out["input_features"]
    inputs["input_features_mask"] = fe_out["input_features_mask"]

    n_ids = int((inputs["input_ids"] == processor.audio_token_id).sum())
    if n_ids != n_soft:
        raise ValueError(
            f"audio placeholder mismatch after build: {n_ids} input_ids vs {n_soft} encoder soft tokens"
        )
    return inputs


def gemma_inputs_to_device(inputs, model):
    import torch

    inputs = inputs.to(model.device)
    for key, val in inputs.items():
        if key in ("input_features", "input_features_mask"):
            continue
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            inputs[key] = val.to(dtype=model.dtype)
    return inputs
