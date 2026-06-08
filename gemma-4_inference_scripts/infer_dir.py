"""
infer_dir.py — Multi-model inference on a local directory of audio files.

Supports five backends:
  gemma4       — google/gemma-4-E2B-it (merged or base+LoRA adapter)
  gemma4_12b   — google/gemma-4-12B-it (base, no fine-tuning)
  gemini       — google/gemini-2.5-flash via OpenRouter (OPENROUTER_API_KEY in .env)
  whisper      — smutuvi/ndizi_whisper_large_v3_turbo_merged_may_26
  w2vbert      — Ndizi fine-tuned badrex/w2v-bert-2.0-swahili-asr (local checkpoint)

No reference transcriptions are required; outputs predictions only (no WER/CER).
Only imports from eval_io.py (no dependency on inference_type_1/2.py).

Usage:
  python gemma-4_inference_scripts/infer_dir.py \\
      --infer-dir data/ndizi-demo_june_3 \\
      --models gemma4 gemini whisper \\
      --gemma4-weights lora \\
      --output-dir results/june3

  # include 12B base Gemma baseline (needs multi-GPU or --device-map auto)
  python gemma-4_inference_scripts/infer_dir.py \\
      --infer-dir data/ndizi-demo_june_3 \\
      --models gemma4 gemma4_12b whisper \\
      --device-map auto

  # debug: 2 files, whisper only
  python gemma-4_inference_scripts/infer_dir.py \\
      --infer-dir data/ndizi-demo_june_3 --models whisper --max-samples 2

  # w2v-bert Ndizi badrex domain checkpoint (server path)
  python gemma-4_inference_scripts/infer_dir.py \\
      --infer-dir /home/smutuvi/data_others \\
      --models w2vbert \\
      --output-dir results/data_others_w2vbert

  # override OpenRouter key at the CLI
  python gemma-4_inference_scripts/infer_dir.py \\
      --infer-dir data/ndizi-demo_june_3 --models gemini --openrouter-key sk-or-...
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly

# ── path setup ────────────────────────────────────────────────────────────────
logging.getLogger("torchao").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_io import (  # noqa: E402
    PUNCTUATION_ASR_INSTRUCTION,
    as_audio_dict,
    collapse_repeated_words,
    gemma_generate_kwargs,
    polish_transcript_casing,
)

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_SR = 16_000
AUDIO_EXTS = {".webm", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Gemma-4 E2B (fine-tuned Ndizi)
MERGED_MODEL_ID = "smutuvi/gemma-4-e2b-sw-asr-ndizi-merged"
PROCESSOR_MODEL_ID = "google/gemma-4-E2B-it"
BASE_MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_REPO = "smutuvi/gemma-4-e2b-sw-asr-ndizi"

# Gemma-4 12B (base, no Ndizi fine-tuning)
GEMMA4_12B_MODEL_ID = "google/gemma-4-12B-it"
CHUNK_LENGTH_S = 30.0
STRIDE_LENGTH_S = None
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 4

# Whisper
WHISPER_MODEL_ID = "smutuvi/ndizi_whisper_large_v3_turbo_merged_may_26"
WHISPER_LANGUAGE = "sw"
WHISPER_TASK = "transcribe"
WHISPER_CHUNK_LENGTH_S = 30.0
WHISPER_STRIDE_LENGTH_S = 6.0

# w2v-BERT (Ndizi badrex domain fine-tune)
W2VBERT_CHECKPOINT = (
    "/home/smutuvi/ndizi_mlops/inprogress/ndizi-w2vbert-badrex-domain-3epoch-qc/"
    "badrex-w2v-bert-2.0-swahili-asr-08062026-055834"
)
W2VBERT_CHUNK_LENGTH_S = 30.0

# Gemini via OpenRouter
GEMINI_MODEL_ID = "google/gemini-2.5-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PUNCTUATION_FIX_PROMPT = (
    "You are a Swahili text editor. Your only job is to add correct punctuation to the "
    "transcription below. Rules:\n"
    "- End every declarative sentence with a period (.).\n"
    "- End every question with a question mark (?).\n"
    "- Use commas (,) to separate clauses, listed items, and after introductory phrases.\n"
    "- Do NOT change, add, or remove any words — only insert or correct punctuation marks.\n"
    "- Output only the corrected transcription text, nothing else.\n\n"
    "Transcription: {text}"
)


# ── audio file loading ────────────────────────────────────────────────────────
def load_audio_ffmpeg(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode any audio format (incl. WebM) to float32 mono PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(target_sr),
        "-loglevel", "error", "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {r.stderr.decode().strip()}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def scan_audio_files(directory: Path, max_samples: int | None) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if max_samples is not None:
        files = files[:max_samples]
    return files


# ── .env / API key resolution ─────────────────────────────────────────────────
def _load_dotenv(*paths: str | Path) -> None:
    """Minimal .env parser — sets os.environ without overwriting existing keys."""
    for env_path in paths:
        p = Path(env_path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
        return


def resolve_openrouter_key(cli_key: str | None) -> str | None:
    if cli_key:
        return cli_key.strip()
    _load_dotenv(".env", Path.cwd() / ".env", SCRIPT_DIR.parent / ".env", SCRIPT_DIR / ".env")
    return os.environ.get("OPENROUTER_API_KEY", "").strip() or None


# ── Gemma audio helpers ───────────────────────────────────────────────────────
def to_mono(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.ascontiguousarray(wav.squeeze())


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = to_mono(wav)
    if int(sr) == TARGET_SR:
        return wav
    return resample_poly(wav, TARGET_SR, int(sr)).astype(np.float32)


def chunk_waveform_16k(wav16: np.ndarray, chunk_length_s: float, stride_length_s=None):
    chunk_samples = max(1, int(chunk_length_s * TARGET_SR))
    if len(wav16) <= chunk_samples:
        return [wav16]
    stride_s = chunk_length_s if stride_length_s is None else float(stride_length_s)
    stride_samples = max(1, int(stride_s * TARGET_SR))
    chunks, start = [], 0
    while start < len(wav16):
        end = min(start + chunk_samples, len(wav16))
        chunks.append(wav16[start:end])
        if end >= len(wav16):
            break
        start += stride_samples
    return chunks


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


def build_inputs(processor, wave16: np.ndarray, instruction: str = PUNCTUATION_ASR_INSTRUCTION):
    fe_out = processor.feature_extractor([wave16], return_tensors="pt")
    n_soft = gemma_soft_token_count(processor, fe_out)
    boa, at, eoa = processor.boa_token, processor.audio_token, processor.eoa_token
    audio_block = f"{boa}{at * n_soft}{eoa}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": wave16},
            {"type": "text", "text": instruction},
        ],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            f"Audio features and audio tokens do not match, tokens: {n_ids}, features: {n_soft}"
        )
    return inputs


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def inputs_to_device(inputs, model, device: torch.device):
    model_dtype = getattr(model, "dtype", torch.bfloat16)
    for key, val in inputs.items():
        if not isinstance(val, torch.Tensor):
            continue
        if val.is_floating_point():
            inputs[key] = val.to(device=device, dtype=model_dtype)
        else:
            inputs[key] = val.to(device=device)
    return inputs


@torch.no_grad()
def transcribe_wave16(
    model,
    processor,
    wave16: np.ndarray,
    device: torch.device,
    *,
    anti_loop: bool = True,
    casing_polish: bool = True,
    instruction: str | None = None,
) -> str:
    inputs = build_inputs(
        processor, wave16,
        instruction if instruction is not None else PUNCTUATION_ASR_INSTRUCTION,
    )
    inputs = inputs_to_device(inputs, model, device)
    gen_kw = gemma_generate_kwargs(
        MAX_NEW_TOKENS,
        anti_loop=anti_loop,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    )
    out = model.generate(**inputs, **gen_kw)
    new_tokens = out[:, inputs["input_ids"].shape[-1]:]
    hyp = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    if anti_loop:
        hyp = collapse_repeated_words(hyp)
    if casing_polish:
        hyp = polish_transcript_casing(hyp)
    return hyp


def transcribe_audio_dict_chunked(
    model,
    processor,
    audio,
    device: torch.device,
    *,
    anti_loop: bool = True,
    casing_polish: bool = True,
    instruction: str | None = None,
) -> str:
    audio_dict = as_audio_dict(audio)
    wav16 = resample_to_16k(audio_dict["array"], audio_dict["sampling_rate"])
    parts = []
    for ch in chunk_waveform_16k(wav16, CHUNK_LENGTH_S, STRIDE_LENGTH_S):
        hyp = transcribe_wave16(
            model, processor, ch, device,
            anti_loop=anti_loop, casing_polish=False, instruction=instruction,
        )
        if hyp:
            parts.append(hyp)
    text = " ".join(parts)
    return polish_transcript_casing(text) if casing_polish else text


# ── HF auth ───────────────────────────────────────────────────────────────────
def _try_colab_userdata_token() -> str | None:
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN").strip()
    except Exception:
        return None


def resolve_hf_token(cli_token: str | None, *, use_colab: bool = True) -> str | None:
    if cli_token:
        return cli_token.strip()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    if use_colab:
        tok = _try_colab_userdata_token()
        if tok:
            print("Using HF_TOKEN from Colab userdata secret")
            return tok
    else:
        try:
            import google.colab  # noqa: F401
            tok = _try_colab_userdata_token()
            if tok:
                print("Using HF_TOKEN from Colab userdata secret")
                return tok
        except ImportError:
            pass
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            print("Using token from huggingface-cli login cache")
        return cached
    except Exception:
        return None


def setup_hf_auth(token: str | None) -> str | None:
    if not token:
        print(
            "WARNING: No Hugging Face token. Private/gated repos will 401.\n"
            "  Pass --hf-token hf_...  or set HF_TOKEN in .env / Colab secrets."
        )
        return None
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    print(f"HF auth OK (token …{token[-4:]})")
    return token


def _assert_model_materialized(model) -> None:
    meta = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        raise RuntimeError(
            f"{len(meta)} parameter(s) still on meta device (e.g. {meta[0]}). "
            "Re-run with --device-map cuda on a GPU runtime."
        )


# ── Gemma model loading ───────────────────────────────────────────────────────
def load_merged_model(dtype: torch.dtype, token: str | None, *, device_map_mode: str = "cuda"):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    hub_kw = {"token": token} if token else {}
    device = resolve_torch_device() if device_map_mode != "cpu" else torch.device("cpu")

    print(f"Loading processor: {PROCESSOR_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(PROCESSOR_MODEL_ID, padding_side="left", **hub_kw)

    print(f"Loading merged weights: {MERGED_MODEL_ID}")
    load_kw: dict = {
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": False,
        **hub_kw,
    }
    if device_map_mode == "auto":
        load_kw["device_map"] = "auto"
    elif device.type == "cuda":
        load_kw["device_map"] = {"": device.index or 0}

    try:
        model = AutoModelForMultimodalLM.from_pretrained(MERGED_MODEL_ID, **load_kw)
        if device_map_mode != "auto" and device.type == "cpu":
            model = model.to(device)
        model = model.eval()
    except OSError as exc:
        raise OSError(
            f"\nFailed to load {MERGED_MODEL_ID}.\n"
            "  1) Accept the Gemma license: https://huggingface.co/google/gemma-4-E2B-it\n"
            "  2) Ensure your HF token has read access to the merged repo.\n"
            "  3) Pass --hf-token hf_...  or try --gemma4-weights lora instead."
        ) from exc

    _assert_model_materialized(model)
    print(f"Inference device: {device}  (device_map={device_map_mode})")
    return model, processor, device


def load_lora_model(dtype: torch.dtype, token: str | None):
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    hub_kw = {"token": token} if token else {}
    print(f"Loading processor: {BASE_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, padding_side="left", **hub_kw)
    print(f"Loading base model: {BASE_MODEL_ID}")
    print(f"Loading LoRA adapter: {ADAPTER_REPO}")
    base = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        **hub_kw,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=token).eval()
    device = resolve_torch_device()
    print(f"Inference device: {device}  (device_map=auto)")
    return model, processor, device


def load_gemma4_12b_model(dtype: torch.dtype, token: str | None, *, device_map_mode: str = "auto"):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    hub_kw = {"token": token} if token else {}
    # 12B in bf16 needs ~24 GB+ VRAM — prefer accelerate sharding over a single GPU index.
    effective_map = "auto" if device_map_mode == "cuda" else device_map_mode
    device = resolve_torch_device() if effective_map != "cpu" else torch.device("cpu")

    print(f"Loading processor: {GEMMA4_12B_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        GEMMA4_12B_MODEL_ID, padding_side="left", **hub_kw
    )

    print(f"Loading base model: {GEMMA4_12B_MODEL_ID}")
    load_kw: dict = {
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": False,
        **hub_kw,
    }
    if effective_map == "auto":
        load_kw["device_map"] = "auto"
    elif device.type == "cuda":
        load_kw["device_map"] = {"": device.index or 0}

    try:
        model = AutoModelForMultimodalLM.from_pretrained(GEMMA4_12B_MODEL_ID, **load_kw)
        if effective_map != "auto" and device.type == "cpu":
            model = model.to(device)
        model = model.eval()
    except OSError as exc:
        raise OSError(
            f"\nFailed to load {GEMMA4_12B_MODEL_ID}.\n"
            "  1) Accept the Gemma license: https://huggingface.co/google/gemma-4-12B-it\n"
            "  2) Pass --hf-token hf_...  or set HF_TOKEN in .env / Colab secrets.\n"
            "  3) Use --device-map auto if a single GPU OOMs."
        ) from exc

    _assert_model_materialized(model)
    print(f"Inference device: {device}  (device_map={effective_map})")
    return model, processor, device


# ── Gemini via OpenRouter ─────────────────────────────────────────────────────
def _encode_wav_base64(arr: np.ndarray, sr: int) -> str:
    import base64
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def transcribe_gemini(arr: np.ndarray, api_key: str) -> str:
    import json as _json
    import urllib.request

    b64 = _encode_wav_base64(arr, TARGET_SR)
    payload = {
        "model": GEMINI_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:audio/wav;base64,{b64}"}},
                {"type": "text", "text": PUNCTUATION_ASR_INSTRUCTION},
            ],
        }],
    }
    body = _json.dumps(payload).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/smutuvi/ndizi",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read().decode())

    text = data["choices"][0]["message"]["content"].strip()
    text = collapse_repeated_words(text)
    return polish_transcript_casing(text)


def fix_punctuation_gemini(text: str, api_key: str) -> str:
    """Send a raw Gemma-4 transcript to Gemini for punctuation restoration (text-only call)."""
    import json as _json
    import urllib.request

    payload = {
        "model": GEMINI_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": PUNCTUATION_FIX_PROMPT.format(text=text),
        }],
    }
    body = _json.dumps(payload).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/smutuvi/ndizi",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = _json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


# ── Whisper backend ───────────────────────────────────────────────────────────
def _pipeline_device_arg(device: torch.device) -> int | str:
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    if device.type == "mps":
        return "mps"
    return -1


def _extract_pipeline_text(result: Any) -> str:
    """Normalise pipeline output to a plain string (handles dict / list / chunk formats)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        text = result.get("text")
        if text is not None:
            return str(text).strip()
        chunks = result.get("chunks")
        if isinstance(chunks, list):
            parts = [str(c.get("text", "")).strip() for c in chunks if isinstance(c, dict)]
            return " ".join(p for p in parts if p)
        return ""
    if isinstance(result, list):
        if not result:
            return ""
        if isinstance(result[0], dict) and "text" in result[0]:
            return " ".join(str(x.get("text", "")).strip() for x in result if isinstance(x, dict))
        if isinstance(result[0], str):
            return " ".join(str(x).strip() for x in result)
        return _extract_pipeline_text(result[0])
    return str(result).strip()


def load_whisper_pipeline(device: torch.device):
    from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading Whisper processor: {WHISPER_MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=WHISPER_LANGUAGE, task=WHISPER_TASK
    )

    print(f"Loading Whisper model: {WHISPER_MODEL_ID}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=_pipeline_device_arg(device),
        chunk_length_s=WHISPER_CHUNK_LENGTH_S,
        stride_length_s=WHISPER_STRIDE_LENGTH_S,
    )
    return pipe, forced_decoder_ids


def transcribe_whisper(pipe, forced_decoder_ids: list, arr: np.ndarray) -> str:
    result = pipe(
        {"array": arr, "sampling_rate": TARGET_SR},
        generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
    )
    text = _extract_pipeline_text(result)
    return polish_transcript_casing(text)


# ── w2v-BERT backend ──────────────────────────────────────────────────────────
def _probe_w2vbert_input_name(processor) -> str:
    dummy = np.zeros(TARGET_SR, dtype=np.float32)
    out = processor(dummy, sampling_rate=TARGET_SR)
    if isinstance(out, dict):
        for key in ("input_features", "input_values"):
            if key in out:
                return key
        raise RuntimeError(f"Processor keys not usable: {list(out.keys())}")
    for key in ("input_features", "input_values"):
        if hasattr(out, key):
            return key
    raise RuntimeError("Processor output has no input_features / input_values")


def _w2vbert_feat_row(processor, model_input_name: str, arr: np.ndarray) -> dict:
    out = processor(arr, sampling_rate=TARGET_SR)
    if isinstance(out, dict):
        feat = out[model_input_name][0]
    else:
        feat = getattr(out, model_input_name)[0]
    return {model_input_name: feat}


def _chunk_waveform_for_w2vbert(arr: np.ndarray, chunk_seconds: float) -> list[np.ndarray]:
    wav = np.asarray(arr, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return []

    chunk_samples = max(1, int(chunk_seconds * TARGET_SR))
    min_standalone = max(4000, int(0.25 * TARGET_SR))

    segs: list[np.ndarray] = []
    for off in range(0, wav.size, chunk_samples):
        segs.append(np.copy(wav[off : off + chunk_samples]))
    while len(segs) >= 2 and segs[-1].size < min_standalone:
        segs[-2] = np.concatenate([segs[-2], segs[-1]])
        segs.pop()

    out: list[np.ndarray] = []
    for seg in segs:
        if seg.size == 0:
            continue
        if seg.size < min_standalone:
            seg = np.pad(seg, (0, min_standalone - int(seg.size)), mode="constant")
        out.append(np.ascontiguousarray(seg, dtype=np.float32))
    return out


def _decode_w2vbert_ids(processor, pred_ids: np.ndarray) -> str:
    arr = np.asarray(pred_ids)
    if arr.ndim == 1:
        arr = arr[None, :]
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(arr)[0].strip()
    return processor.tokenizer.batch_decode(arr)[0].strip()


@torch.no_grad()
def _transcribe_w2vbert_segment(
    model,
    processor,
    model_input_name: str,
    arr: np.ndarray,
    device: torch.device,
) -> str:
    feat = _w2vbert_feat_row(processor, model_input_name, arr)
    batch = processor.pad([feat], padding=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
    logits = model(**batch).logits
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    return _decode_w2vbert_ids(processor, pred_ids)


def load_w2vbert_model(checkpoint: str, device: torch.device):
    from transformers import AutoModelForCTC, AutoProcessor

    ckpt = Path(checkpoint).expanduser()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"w2v-bert checkpoint not found: {ckpt}")

    print(f"Loading w2v-bert processor: {ckpt}")
    processor = AutoProcessor.from_pretrained(str(ckpt))
    if getattr(processor, "tokenizer", None) is None and not hasattr(processor, "batch_decode"):
        raise RuntimeError("w2v-bert processor has no tokenizer; cannot decode CTC outputs.")

    model_input_name = _probe_w2vbert_input_name(processor)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading w2v-bert model: {ckpt}")
    model = AutoModelForCTC.from_pretrained(str(ckpt), torch_dtype=dtype)
    model = model.to(device).eval()
    print(f"Inference device: {device}  model_input_name={model_input_name}")
    return model, processor, model_input_name


def transcribe_w2vbert(
    model,
    processor,
    model_input_name: str,
    arr: np.ndarray,
    device: torch.device,
    *,
    chunk_length_s: float = W2VBERT_CHUNK_LENGTH_S,
) -> str:
    dur = len(arr) / TARGET_SR
    if dur <= chunk_length_s:
        text = _transcribe_w2vbert_segment(model, processor, model_input_name, arr, device)
        return polish_transcript_casing(text)

    parts = [
        _transcribe_w2vbert_segment(model, processor, model_input_name, seg, device)
        for seg in _chunk_waveform_for_w2vbert(arr, chunk_length_s)
    ]
    text = " ".join(p for p in parts if p)
    return polish_transcript_casing(text)


# ── output helpers ────────────────────────────────────────────────────────────
def _write_model_csv(output_dir: Path, model_name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"predictions_{model_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {csv_path.name}")


def _write_combined_predictions_json(
    output_dir: Path, combined: dict[str, dict[str, Any]]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.json"

    ordered_rows = []
    for row in combined.values():
        meta_keys = [k for k in row if not k.startswith(("prediction_", "rtfx_"))]
        pred_keys = sorted(k for k in row if k.startswith("prediction_"))
        rtfx_keys = sorted(k for k in row if k.startswith("rtfx_"))
        ordered_rows.append({k: row[k] for k in meta_keys + pred_keys + rtfx_keys})

    path.write_text(
        json.dumps(ordered_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def build_metrics_summary(
    model_results: dict[str, list[dict[str, Any]]],
    *,
    infer_dir: str,
    models_run: list[str],
) -> dict[str, Any]:
    model_stats: dict[str, Any] = {}
    for name, rows in model_results.items():
        decode_times = [r["decode_wall_s"] for r in rows if r.get("decode_wall_s") is not None]
        rtfxs = [r["rtfx"] for r in rows if r.get("rtfx") is not None]
        model_stats[name] = {
            "n": len(rows),
            "avg_decode_wall_s": round(sum(decode_times) / len(decode_times), 3) if decode_times else None,
            "avg_rtfx": round(sum(rtfxs) / len(rtfxs), 3) if rtfxs else None,
        }
    return {
        "mode": "inference_only",
        "infer_dir": infer_dir,
        "n_files": max((v["n"] for v in model_stats.values()), default=0),
        "models": model_stats,
        "run_info": {
            "models_run": models_run,
            "infer_dir": infer_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--infer-dir", type=Path, required=True,
        help="Directory containing audio files (webm/wav/mp3/flac/ogg/m4a)",
    )
    p.add_argument(
        "--models", nargs="+",
        choices=("gemma4", "gemma4_12b", "gemini", "whisper", "w2vbert"),
        default=["gemma4", "gemini", "whisper"],
        help="Backends to run (default: gemma4, gemini, whisper; add gemma4_12b or w2vbert)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("infer_dir_results"),
        help="Output directory (default: ./infer_dir_results)",
    )
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap number of audio files for debugging")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token for gated/private models")
    p.add_argument("--openrouter-key", type=str, default=None,
                   help="OpenRouter API key (overrides OPENROUTER_API_KEY in .env)")
    p.add_argument(
        "--gemma4-weights", choices=("merged", "lora"), default="merged",
        help="merged = smutuvi/gemma-4-e2b-sw-asr-ndizi-merged (default); "
             "lora = google/gemma-4-E2B-it + smutuvi/gemma-4-e2b-sw-asr-ndizi adapter",
    )
    p.add_argument(
        "--device-map", choices=("cuda", "auto", "cpu"), default="cuda",
        help="Device placement for Gemma/Whisper/w2v-bert (default: cuda)",
    )
    p.add_argument(
        "--w2vbert-checkpoint", type=str, default=W2VBERT_CHECKPOINT,
        help=f"Local w2v-bert checkpoint directory (default: {W2VBERT_CHECKPOINT})",
    )
    p.add_argument("--gemma4-punctuation-fix", action="store_true",
                   help="Post-process Gemma-4 transcriptions with Gemini to restore punctuation "
                        "(requires --openrouter-key or OPENROUTER_API_KEY)")
    p.add_argument("--no-anti-loop", action="store_true",
                   help="Disable repetition penalty / stutter collapse for Gemma")
    p.add_argument("--no-casing-polish", action="store_true",
                   help="Skip sentence-start capitalization for Gemma")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if not args.infer_dir.is_dir():
        raise SystemExit(f"--infer-dir {args.infer_dir} is not a directory")

    audio_files = scan_audio_files(args.infer_dir, args.max_samples)
    if not audio_files:
        raise SystemExit(
            f"No audio files ({', '.join(sorted(AUDIO_EXTS))}) found in {args.infer_dir}"
        )
    print(f"Found {len(audio_files)} audio file(s) in {args.infer_dir}")

    anti_loop = not args.no_anti_loop
    casing_polish = not args.no_casing_polish

    if not torch.cuda.is_available() and args.device_map == "cuda":
        print("No CUDA GPU detected; falling back to CPU")
        args.device_map = "cpu"
    device = torch.device(
        "cpu" if args.device_map == "cpu" else
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )

    print("\nDecoding audio files...")
    audio_arrays: list[tuple[str, np.ndarray, float]] = []
    for p in audio_files:
        arr = load_audio_ffmpeg(p)
        dur = len(arr) / TARGET_SR
        audio_arrays.append((p.name, arr, dur))
        print(f"  {p.name:22s}  {dur:.1f}s")

    model_results: dict[str, list[dict[str, Any]]] = {}
    combined: dict[str, dict[str, Any]] = {
        fname: {"audio_file": fname, "audio_duration_s": round(dur, 3)}
        for fname, _, dur in audio_arrays
    }

    # ── gemma4 ────────────────────────────────────────────────────────────────
    if "gemma4" in args.models:
        print(f"\n{'='*60}\ngemma4 backend ({args.gemma4_weights} weights)\n{'='*60}")
        token = setup_hf_auth(resolve_hf_token(args.hf_token, use_colab=True))
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if args.gemma4_weights == "lora":
            model, processor, inf_device = load_lora_model(dtype, token)
        else:
            model, processor, inf_device = load_merged_model(
                dtype, token, device_map_mode=args.device_map
            )

        punct_fix_key: str | None = None
        if args.gemma4_punctuation_fix:
            punct_fix_key = resolve_openrouter_key(args.openrouter_key)
            if not punct_fix_key:
                print("WARNING: --gemma4-punctuation-fix requires an OpenRouter key — fix skipped.\n"
                      "  Pass --openrouter-key or set OPENROUTER_API_KEY in .env")

        rows: list[dict[str, Any]] = []
        for fname, arr, dur in audio_arrays:
            print(f"  {fname}", end=" ... ", flush=True)
            t0 = time.perf_counter()
            hyp = transcribe_audio_dict_chunked(
                model, processor,
                {"array": arr, "sampling_rate": TARGET_SR},
                inf_device,
                anti_loop=anti_loop,
                casing_polish=casing_polish,
                instruction=PUNCTUATION_ASR_INSTRUCTION,
            )
            wall = time.perf_counter() - t0
            rtfx = round(dur / wall, 3) if wall > 0 else None
            if punct_fix_key:
                try:
                    hyp = fix_punctuation_gemini(hyp, punct_fix_key)
                except Exception as exc:
                    print(f"\n    punctuation fix ERROR: {exc}")
            print(f"{wall:.1f}s  RTFx={rtfx}" if rtfx else f"{wall:.1f}s")
            rows.append({"audio_file": fname, "prediction": hyp,
                          "audio_duration_s": round(dur, 3),
                          "decode_wall_s": round(wall, 3), "rtfx": rtfx})
            combined[fname]["prediction_gemma4"] = hyp
            combined[fname]["rtfx_gemma4"] = rtfx
        model_results["gemma4"] = rows
        _write_model_csv(args.output_dir, "gemma4", rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── gemma4_12b ────────────────────────────────────────────────────────────
    if "gemma4_12b" in args.models:
        print(f"\n{'='*60}\ngemma4_12b backend ({GEMMA4_12B_MODEL_ID})\n{'='*60}")
        token = setup_hf_auth(resolve_hf_token(args.hf_token, use_colab=True))
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model, processor, inf_device = load_gemma4_12b_model(
            dtype, token, device_map_mode=args.device_map
        )

        rows = []
        for fname, arr, dur in audio_arrays:
            print(f"  {fname}", end=" ... ", flush=True)
            t0 = time.perf_counter()
            hyp = transcribe_audio_dict_chunked(
                model, processor,
                {"array": arr, "sampling_rate": TARGET_SR},
                inf_device,
                anti_loop=anti_loop,
                casing_polish=casing_polish,
                instruction=PUNCTUATION_ASR_INSTRUCTION,
            )
            wall = time.perf_counter() - t0
            rtfx = round(dur / wall, 3) if wall > 0 else None
            print(f"{wall:.1f}s  RTFx={rtfx}" if rtfx else f"{wall:.1f}s")
            rows.append({"audio_file": fname, "prediction": hyp,
                          "audio_duration_s": round(dur, 3),
                          "decode_wall_s": round(wall, 3), "rtfx": rtfx})
            combined[fname]["prediction_gemma4_12b"] = hyp
            combined[fname]["rtfx_gemma4_12b"] = rtfx
        model_results["gemma4_12b"] = rows
        _write_model_csv(args.output_dir, "gemma4_12b", rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── gemini ────────────────────────────────────────────────────────────────
    if "gemini" in args.models:
        print(f"\n{'='*60}\ngemini backend ({GEMINI_MODEL_ID} via OpenRouter)\n{'='*60}")
        api_key = resolve_openrouter_key(args.openrouter_key)
        if not api_key:
            print(
                "WARNING: No OPENROUTER_API_KEY found — skipping gemini backend.\n"
                "  Add OPENROUTER_API_KEY=sk-or-... to a .env file, or pass --openrouter-key"
            )
        else:
            rows = []
            for fname, arr, dur in audio_arrays:
                print(f"  {fname}", end=" ... ", flush=True)
                t0 = time.perf_counter()
                try:
                    hyp = transcribe_gemini(arr, api_key)
                except Exception as exc:
                    print(f"\n    ERROR: {exc}")
                    hyp = f"[ERROR: {exc}]"
                wall = time.perf_counter() - t0
                print(f"{wall:.1f}s")
                rows.append({"audio_file": fname, "prediction": hyp,
                              "audio_duration_s": round(dur, 3),
                              "decode_wall_s": round(wall, 3), "rtfx": None})
                combined[fname]["prediction_gemini"] = hyp
                combined[fname]["rtfx_gemini"] = None
            model_results["gemini"] = rows
            _write_model_csv(args.output_dir, "gemini", rows)

    # ── whisper ───────────────────────────────────────────────────────────────
    if "whisper" in args.models:
        print(f"\n{'='*60}\nwhisper backend\n{'='*60}")
        setup_hf_auth(resolve_hf_token(args.hf_token, use_colab=True))
        pipe, forced_decoder_ids = load_whisper_pipeline(device)
        rows = []
        for fname, arr, dur in audio_arrays:
            print(f"  {fname}", end=" ... ", flush=True)
            t0 = time.perf_counter()
            hyp = transcribe_whisper(pipe, forced_decoder_ids, arr)
            wall = time.perf_counter() - t0
            rtfx = round(dur / wall, 3) if wall > 0 else None
            print(f"{wall:.1f}s  RTFx={rtfx}" if rtfx else f"{wall:.1f}s")
            rows.append({"audio_file": fname, "prediction": hyp,
                          "audio_duration_s": round(dur, 3),
                          "decode_wall_s": round(wall, 3), "rtfx": rtfx})
            combined[fname]["prediction_whisper"] = hyp
            combined[fname]["rtfx_whisper"] = rtfx
        model_results["whisper"] = rows
        _write_model_csv(args.output_dir, "whisper", rows)

    # ── w2vbert ───────────────────────────────────────────────────────────────
    if "w2vbert" in args.models:
        print(f"\n{'='*60}\nw2vbert backend\n{'='*60}")
        model, processor, model_input_name = load_w2vbert_model(args.w2vbert_checkpoint, device)
        rows = []
        for fname, arr, dur in audio_arrays:
            print(f"  {fname}", end=" ... ", flush=True)
            t0 = time.perf_counter()
            hyp = transcribe_w2vbert(
                model, processor, model_input_name, arr, device,
                chunk_length_s=W2VBERT_CHUNK_LENGTH_S,
            )
            wall = time.perf_counter() - t0
            rtfx = round(dur / wall, 3) if wall > 0 else None
            print(f"{wall:.1f}s  RTFx={rtfx}" if rtfx else f"{wall:.1f}s")
            rows.append({"audio_file": fname, "prediction": hyp,
                          "audio_duration_s": round(dur, 3),
                          "decode_wall_s": round(wall, 3), "rtfx": rtfx})
            combined[fname]["prediction_w2vbert"] = hyp
            combined[fname]["rtfx_w2vbert"] = rtfx
        model_results["w2vbert"] = rows
        _write_model_csv(args.output_dir, "w2vbert", rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── write outputs ─────────────────────────────────────────────────────────
    if not model_results:
        print("\nNo models ran — nothing to write.")
        return

    preds_path = _write_combined_predictions_json(args.output_dir, combined)

    metrics = build_metrics_summary(
        model_results,
        infer_dir=str(args.infer_dir.resolve()),
        models_run=list(model_results.keys()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, stats in metrics["models"].items():
        rtfx_str = f"  avg_RTFx={stats['avg_rtfx']}" if stats["avg_rtfx"] else ""
        print(f"  {name:8s}  n={stats['n']}  avg_decode={stats['avg_decode_wall_s']}s{rtfx_str}")
    print(f"\nWrote {preds_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
