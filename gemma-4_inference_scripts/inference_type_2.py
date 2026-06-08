"""
Eval script aligned with inference_type_1.py (chunking, full test splits).

Only difference: loads merged Hub weights (type 1 default is base + LoRA adapter).

Colab (GPU runtime):
  1. Secrets → add HF_TOKEN (read token; accept Gemma license on google/gemma-4-E2B-it)
  2. Upload this file or clone the repo, then either:
       %run inference_type_2.py --max-samples 8          # in-notebook (sees Colab secret)
       !python inference_type_2.py --colab --max-samples 8
  3. Full eval: %run inference_type_2.py
  4. Writes metrics.json, predictions.json, predictions.csv under --output-dir

Usage:
  python additional_scripts/inference_type_2.py
  python additional_scripts/inference_type_2.py --normalize jiwer_default
  python additional_scripts/inference_type_2.py --output-dir eval/gemma4-merged-run --normalize jiwer_default
  python additional_scripts/inference_type_2.py --hf-token hf_xxxx --output-dir eval/gemma4-merged-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

# Non-fatal torchao extension warnings on Colab (wrong ABI / Python version).
logging.getLogger("torchao").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_io import (  # noqa: E402
    DEFAULT_ASR_INSTRUCTION,
    as_audio_dict,
    build_metrics_payload,
    build_prediction_rows,
    collect_audio_path_labels,
    collapse_repeated_words,
    eval_row_meta,
    gemma_generate_kwargs,
    polish_transcript_casing,
    pooled_wer_cer,
    resolve_eval_columns,
    write_eval_outputs,
)

# =============================================================================
# CONFIG — merged weights only (contrast with type_1 base + LoRA)
# =============================================================================
MERGED_MODEL_ID = "smutuvi/gemma-4-e2b-sw-asr-ndizi-merged"
# Processor always from base Gemma (public/gated once); merged repo may be private or
# lack processor_config.json — weights still load from MERGED_MODEL_ID.
PROCESSOR_MODEL_ID = "google/gemma-4-E2B-it"
DATASET_SPECS = [
    ("smutuvi/ndizi-1", "test"),
    ("smutuvi/ndizi-1-2025", "test"),
]
CHUNK_LENGTH_S = 30.0
STRIDE_LENGTH_S = None  # no overlap (same as repo default)
MAX_NEW_TOKENS = 128
TARGET_SR = 16_000
# Defaults match scripts/evaluate_gemma4.py --anti-loop-decode
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 4
ASR_INSTRUCTION = DEFAULT_ASR_INSTRUCTION

# =============================================================================
# Audio + Gemma multimodal inputs (audio token alignment)
# =============================================================================
def to_mono(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.ascontiguousarray(wav.squeeze())


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    sr = int(sr)
    wav = to_mono(wav)
    if sr == TARGET_SR:
        return wav
    return resample_poly(wav, TARGET_SR, sr).astype(np.float32)


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


def build_inputs(processor, wave16: np.ndarray, instruction=ASR_INSTRUCTION):
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
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
            f"Audio features and audio tokens do not match, tokens: {n_ids}, features: {n_soft}"
        )
    return inputs


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def inputs_to_device(inputs, model, device: torch.device):
    """Move all multimodal tensors to the inference device (incl. input_features on GPU)."""
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
    inputs = build_inputs(processor, wave16, instruction if instruction is not None else ASR_INSTRUCTION)
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
            model, processor, ch, device, anti_loop=anti_loop, casing_polish=False,
            instruction=instruction,
        )
        if hyp:
            parts.append(hyp)
    text = " ".join(parts)
    return polish_transcript_casing(text) if casing_polish else text


def get_text(row) -> str:
    for k in ("text", "sentence", "transcript", "normalized_text"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def _try_colab_userdata_token() -> str | None:
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN").strip()
    except Exception:
        return None


def resolve_hf_token(cli_token: str | None, *, use_colab: bool) -> str | None:
    """--hf-token → env → Colab secret → huggingface-cli cache."""
    if cli_token:
        return cli_token.strip()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    if use_colab:
        colab_tok = _try_colab_userdata_token()
        if colab_tok:
            print("Using HF_TOKEN from Colab userdata secret")
            return colab_tok
    else:
        # Auto-detect Colab when not explicitly disabled (running in notebook)
        try:
            import google.colab  # noqa: F401

            colab_tok = _try_colab_userdata_token()
            if colab_tok:
                print("Using HF_TOKEN from Colab userdata secret")
                return colab_tok
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
    """Register token with huggingface_hub and mirror into env for subprocess libraries."""
    if not token:
        print(
            "WARNING: No Hugging Face token. Private/gated repos will 401.\n"
            "  Colab: python inference_type_2.py --hf-token $(python -c "
            "\"from google.colab import userdata; print(userdata.get('HF_TOKEN'), end='')\")\n"
            "  Or: huggingface-cli login  OR  export HF_TOKEN=hf_..."
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


def load_merged_model(
    dtype: torch.dtype,
    token: str | None,
    *,
    processor_model_id: str = PROCESSOR_MODEL_ID,
    device_map_mode: str = "cuda",
):
    """Processor from base Gemma; finetuned weights from merged Hub repo."""
    hub_kw = {"token": token} if token else {}
    if device_map_mode == "cpu":
        device = torch.device("cpu")
    else:
        device = resolve_torch_device()

    print(f"Loading processor: {processor_model_id}")
    processor = AutoProcessor.from_pretrained(
        processor_model_id, padding_side="left", **hub_kw
    )
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
        # Single-GPU Colab: avoid meta/CPU mismatch from device_map="auto" on Gemma4 multimodal.
        load_kw["device_map"] = {"": device.index or 0}
    else:
        pass  # load on CPU, then .to(device)

    try:
        model = AutoModelForMultimodalLM.from_pretrained(MERGED_MODEL_ID, **load_kw)
        if device_map_mode != "auto" and device.type == "cpu":
            model = model.to(device)
        model = model.eval()
    except OSError as exc:
        hint = (
            f"\nFailed to load {MERGED_MODEL_ID}.\n"
            "  1) Accept the Gemma license: https://huggingface.co/google/gemma-4-E2B-it\n"
            "  2) Ensure your HF account can read the merged repo (Settings → Access tokens, read)\n"
            "  3) Pass a valid token: --hf-token hf_...  (Colab secrets often need this flag)\n"
            "  4) If merged stays private, use inference_type_1.py (base + LoRA adapter) instead."
        )
        raise OSError(hint) from exc

    _assert_model_materialized(model)
    print(f"Inference device: {device}  (device_map={device_map_mode})")
    return model, processor, device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--normalize",
        choices=("none", "jiwer_default"),
        default="none",
        help="WER/CER text normalization (default: none = raw jiwer)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "eval_type_2",
        help="Write metrics.json, predictions.json, predictions.csv here (ndizi_mlops layout)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap rows per dataset (debug); default = full test split",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (overrides HF_TOKEN env). Required on Colab for gated/private repos.",
    )
    p.add_argument(
        "--processor-model-id",
        type=str,
        default=PROCESSOR_MODEL_ID,
        help=f"Where to load AutoProcessor (default: {PROCESSOR_MODEL_ID})",
    )
    p.add_argument(
        "--colab",
        action="store_true",
        help="Read HF_TOKEN from Colab userdata secret (also auto-detected in Colab)",
    )
    p.add_argument(
        "--device-map",
        choices=("cuda", "auto", "cpu"),
        default="cuda",
        help="Model placement: cuda=single GPU (default, fixes meta-device errors); auto=accelerate",
    )
    p.add_argument(
        "--no-anti-loop",
        action="store_true",
        help="Disable repetition_penalty / no_repeat_ngram_size / stutter collapse",
    )
    p.add_argument(
        "--no-casing-polish",
        action="store_true",
        help="Skip light sentence-start capitalization after decode",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    normalize = args.normalize
    max_samples = args.max_samples
    anti_loop = not args.no_anti_loop
    casing_polish = not args.no_casing_polish

    token = setup_hf_auth(resolve_hf_token(args.hf_token, use_colab=args.colab))
    if token is None:
        raise SystemExit(
            "No Hugging Face token. On Colab: Secrets → HF_TOKEN, then re-run.\n"
            "Or: python inference_type_2.py --hf-token hf_xxxx"
        )
    if not torch.cuda.is_available() and args.device_map == "cuda":
        print("No CUDA GPU detected; falling back to CPU load")
        args.device_map = "cpu"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor_model_id = args.processor_model_id
    model, processor, device = load_merged_model(
        dtype,
        token,
        processor_model_id=processor_model_id,
        device_map_mode=args.device_map,
    )

    per_set_scores: dict[str, dict] = {}
    all_refs: list[str] = []
    all_hyps: list[str] = []
    prediction_rows: list[dict] = []

    for ds_id, split in DATASET_SPECS:
        key = f"{ds_id}:{split}"
        print(f"\n{'=' * 60}")
        print(f"Evaluating {key}")
        print(f"{'=' * 60}")

        ds = load_dataset(ds_id, split=split)
        audio_col, text_col = resolve_eval_columns(list(ds.column_names))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        path_labels = collect_audio_path_labels(ds, audio_col)
        ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

        refs, hyps, row_meta = [], [], []
        for i, row in enumerate(tqdm(ds, desc=key)):
            ref = str(row.get(text_col) or "")
            hyp = transcribe_audio_dict_chunked(
                model,
                processor,
                row[audio_col],
                device,
                anti_loop=anti_loop,
                casing_polish=casing_polish,
            )
            refs.append(ref)
            hyps.append(hyp)
            ap = path_labels[i] if i < len(path_labels) else f"row_{i}"
            row_meta.append(eval_row_meta(row, row_idx=i, audio_col=audio_col, audio_path=ap))

        wer, cer = pooled_wer_cer(hyps, refs, "none")
        per_set_scores[key] = {
            "wer": wer,
            "cer": cer,
            "n": len(refs),
            "refs": refs,
            "hyps": hyps,
            "source_dataset": ds_id,
        }
        prediction_rows.extend(
            build_prediction_rows(
                dataset=ds_id,
                split=split,
                refs=refs,
                hyps=hyps,
                row_meta=row_meta,
                normalize=normalize,
            )
        )
        all_refs.extend(refs)
        all_hyps.extend(hyps)

        print(f"  n   = {len(refs)}")
        print(f"  WER = {wer:.4f}")
        print(f"  CER = {cer:.4f}")
        if normalize != "none":
            wn, cn = pooled_wer_cer(hyps, refs, normalize)
            print(f"  WER ({normalize}) = {wn:.4f}")
            print(f"  CER ({normalize}) = {cn:.4f}")

    run_info = {
        "script": "inference_type_2.py",
        "model": MERGED_MODEL_ID,
        "processor": processor_model_id,
        "device": str(device),
        "device_map": args.device_map,
        "test_datasets": [f"{d}:{s}" for d, s in DATASET_SPECS],
        "output_dir": str(args.output_dir.resolve()),
        "chunk_length_s": CHUNK_LENGTH_S,
        "stride_length_s": STRIDE_LENGTH_S,
        "max_samples": max_samples,
        "max_new_tokens": MAX_NEW_TOKENS,
        "anti_loop": anti_loop,
        "repetition_penalty": REPETITION_PENALTY if anti_loop else None,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE if anti_loop else None,
    }
    metrics_payload = build_metrics_payload(
        text_normalize=normalize,
        per_set_scores=per_set_scores,
        all_refs=all_refs,
        all_hyps=all_hyps,
        chunk_length_s=CHUNK_LENGTH_S,
        run_info=run_info,
    )
    metrics_path, preds_json_path, preds_csv_path = write_eval_outputs(
        args.output_dir,
        metrics_payload=metrics_payload,
        prediction_rows=prediction_rows,
    )

    pooled = metrics_payload["pooled"]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY (merged weights, text_normalize={normalize})")
    print(f"{'=' * 60}")
    print(
        f"  pooled  n={pooled['n_utterances']}  WER={pooled['wer']:.4f}  CER={pooled['cer']:.4f}"
    )
    if normalize != "none" and "wer_normalized" in pooled:
        print(
            f"          WER_norm={pooled['wer_normalized']:.4f}  "
            f"CER_norm={pooled['cer_normalized']:.4f}"
        )
    for key, block in metrics_payload["per_set"].items():
        print(f"  {key}  n={block['n']}  WER={block['wer']:.4f}  CER={block['cer']:.4f}")
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {preds_json_path}")
    print(f"Wrote {preds_csv_path}")


if __name__ == "__main__":
    main()
