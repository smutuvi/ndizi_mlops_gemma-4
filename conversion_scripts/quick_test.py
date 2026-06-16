"""
quick_test.py
─────────────
Evaluate a .litertlm bundle: Chat and ASR tests via the litert-lm Python API.

Requires (conda ndizi):
  pip install litert-lm-api-nightly

Usage:
  # Sample N clips from HF test sets (default, both ASR + chat):
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model artifacts/bundles/scale07/gemma-4-e2b-sw-asr-ndizi-scale07.litertlm \\
    --n 3

  # Single audio file:
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model /path/to/model.litertlm \\
    --audio /path/to/recording.wav

  # Chat only (skip ASR):
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model /path/to/model.litertlm --no-asr

  # ASR only (skip chat):
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model /path/to/model.litertlm --no-chat --n 3
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = (
    "/home/smutuvi/ndizi_mlops_gemma-4/artifacts/litert_slim"
    "/gemma-4-e2b-sw-asr-ndizi-slim.litertlm"
)
DEFAULT_DATASETS = ["smutuvi/ndizi-1:test", "smutuvi/ndizi-1-2025:test"]
DEFAULT_N = 3

CHAT_PROMPTS = [
    "Habari yako?",
    "Neno 'ndizi' linamaanisha nini?",
    "Nipe muhtasari mfupi wa historia ya Kiswahili.",
]

ASR_INSTRUCTION = "Andika maneno unayosikia katika sauti hii."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_audio_tmp(audio: dict) -> Path:
    """Save a datasets-style audio dict to a 16 kHz int16 WAV (litert_lm compatible)."""
    import numpy as np

    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio.get("sampling_rate", 16000))

    # Resample to 16 kHz — litert_lm audio encoder expects 16 kHz
    if sr != 16000:
        try:
            import torch
            import torchaudio.functional as taf
            arr = taf.resample(torch.from_numpy(arr), sr, 16000).numpy().astype(np.float32)
        except Exception:
            from scipy.signal import resample_poly
            arr = resample_poly(arr, 16000, sr).astype(np.float32)
        sr = 16000

    # Convert float32 → int16 PCM (universally supported by audio decoders)
    arr_int16 = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        import soundfile as sf
        sf.write(str(tmp_path), arr_int16, sr, subtype="PCM_16")
    except ImportError:
        import torch
        import torchaudio
        torchaudio.save(str(tmp_path), torch.from_numpy(arr_int16).unsqueeze(0), sr,
                        encoding="PCM_S", bits_per_sample=16)

    return tmp_path


# ── Chat test ─────────────────────────────────────────────────────────────────

def test_chat(model: str, prompts: list[str]) -> bool:
    print(f"\n{'═'*60}")
    print("CHAT TEST")
    print(f"{'═'*60}")
    try:
        import litert_lm

        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        with litert_lm.Engine(model, backend=litert_lm.Backend.CPU()) as engine:
            with engine.create_conversation(messages=[
                {"role": "system", "content": [{"type": "text",
                 "text": "Wewe ni msaidizi wa lugha ya Kiswahili."}]}
            ]) as conv:
                for i, prompt in enumerate(prompts, 1):
                    print(f"\n  [{i}] User : {prompt}")
                    resp = conv.send_message(prompt)
                    text = resp["content"][0]["text"]
                    print(f"       Bot  : {text}")

        print("\n  ✓ Chat passed")
        return True

    except Exception as e:
        print(f"\n  ✗ Chat failed: {e}")
        return False


# ── ASR test (single file) ────────────────────────────────────────────────────

def _transcribe_file(engine, audio_path: Path) -> str:
    with engine.create_conversation() as conv:
        resp = conv.send_message({
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
                {"type": "text",  "text": ASR_INSTRUCTION},
            ],
        })
    return resp["content"][0]["text"]


def test_asr_file(model: str, audio: str) -> bool:
    print(f"\n{'═'*60}")
    print("ASR TEST — single file")
    print(f"{'═'*60}")
    audio_path = Path(audio)
    if not audio_path.exists():
        print(f"  ✗ Audio file not found: {audio}")
        return False
    try:
        import litert_lm
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        with litert_lm.Engine(model, backend=litert_lm.Backend.CPU(),
                               audio_backend=litert_lm.Backend.CPU()) as engine:
            transcript = _transcribe_file(engine, audio_path)
        print(f"  Transcript: {transcript}")
        print("  ✓ ASR passed")
        return True
    except Exception as e:
        print(f"  ✗ ASR failed: {e}")
        return False


# ── ASR test (datasets) ───────────────────────────────────────────────────────

def test_asr_datasets(model: str, datasets: list[str], n: int) -> bool:
    print(f"\n{'═'*60}")
    print("ASR TEST — dataset samples")
    print(f"{'═'*60}")
    try:
        import datasets as hf_datasets
        import litert_lm

        # ── Phase 1: load all audio and save to temp files BEFORE opening Engine ──
        # Loading datasets/soundfile/torchaudio C++ libs inside the Engine context
        # causes allocator conflicts (malloc corruption). Pre-save everything first.
        samples: list[tuple[Path, str, str]] = []  # (tmp_wav, ref, dataset_label)
        for spec in datasets:
            repo, _, split = spec.partition(":")
            split = split or "test"
            label = f"{repo} [{split}]"
            print(f"\n  Loading: {label}  n={n}")
            ds = hf_datasets.load_dataset(repo, split=f"{split}[:{n}]")
            for row in ds:
                tmp = _save_audio_tmp(row["audio"])
                ref = row.get("text", row.get("sentence", ""))
                samples.append((tmp, ref, label))
        print(f"\n  Saved {len(samples)} audio clips to temp files — opening Engine...")

        # ── Phase 2: open Engine once and transcribe all pre-saved files ──
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        try:
            with litert_lm.Engine(model, backend=litert_lm.Backend.CPU(),
                                   audio_backend=litert_lm.Backend.CPU()) as engine:
                current_label = None
                idx = 0
                for tmp, ref, label in samples:
                    if label != current_label:
                        print(f"\n  Dataset: {label}")
                        print(f"  {'─'*50}")
                        current_label = label
                        idx = 0
                    idx += 1
                    hyp = _transcribe_file(engine, tmp)
                    print(f"\n  Sample {idx}")
                    print(f"    REF: {ref}")
                    print(f"    HYP: {hyp}")
        finally:
            for tmp, _, _ in samples:
                tmp.unlink(missing_ok=True)

        print("\n  ✓ ASR dataset test passed")
        return True

    except Exception as e:
        print(f"\n  ✗ ASR dataset test failed: {e}")
        if "TF_LITE_AUDIO_ENCODER_HW" in str(e):
            print(
                "\n  Hint: audio encoder missing from bundle.\n"
                "  Rebuild with: python scripts/build_litert_lm_slim.py --merged-model ..."
            )
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a .litertlm bundle: Chat + ASR tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Path to .litertlm file")
    ap.add_argument("--audio", default=None,
                    help="Single .wav file for ASR test (skips dataset sampling)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="HF dataset splits (format: repo:split)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="ASR samples per dataset (default: 3)")
    ap.add_argument("--chat-prompts", nargs="+", default=None,
                    help="Custom chat prompts (default: 3 built-in Swahili prompts)")
    ap.add_argument("--no-chat", action="store_true", help="Skip chat test")
    ap.add_argument("--no-asr",  action="store_true", help="Skip ASR test")
    args = ap.parse_args()

    model = str(Path(args.model).resolve())
    if not Path(model).exists():
        print(f"ERROR: model not found: {model}")
        return 1

    print(f"Model : {model}")
    size_gb = Path(model).stat().st_size / (1024 ** 3)
    print(f"Size  : {size_gb:.2f} GB")

    chat_ok = True
    asr_ok  = True
    prompts = args.chat_prompts or CHAT_PROMPTS

    if not args.no_chat:
        chat_ok = test_chat(model, prompts)

    if not args.no_asr:
        if args.audio:
            asr_ok = test_asr_file(model, args.audio)
        else:
            asr_ok = test_asr_datasets(model, args.datasets, args.n)

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    if not args.no_chat:
        print(f"  Chat : {'✓ passed' if chat_ok else '✗ failed'}")
    if not args.no_asr:
        print(f"  ASR  : {'✓ passed' if asr_ok  else '✗ failed'}")

    return 0 if (chat_ok and asr_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
