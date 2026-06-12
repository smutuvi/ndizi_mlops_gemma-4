"""
quick_test.py
─────────────
Evaluate a .litertlm bundle: Chat and ASR tests via the litert-lm Python API.

Requires (conda ndizi):
  pip install litert-lm-api-nightly

Usage:
  # With the slim bundle (recommended):
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model /home/smutuvi/ndizi_mlops_gemma-4/artifacts/litert_slim/gemma-4-e2b-sw-asr-ndizi-slim.litertlm \\
    --audio /path/to/recording.wav

  # With the original 4.7 GB bundle:
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model /home/smutuvi/ndizi_mlops_gemma-4/ndizi_gemma4_litertlm/model.litertlm \\
    --audio /path/to/recording.wav

  # Chat only (skip ASR):
  conda run -n ndizi python conversion_scripts/quick_test.py \\
    --model /path/to/model.litertlm --no-asr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Default paths (server) ────────────────────────────────────────────────────
DEFAULT_MODEL = (
    "/home/smutuvi/ndizi_mlops_gemma-4/artifacts/litert_slim"
    "/gemma-4-e2b-sw-asr-ndizi-slim.litertlm"
)
DEFAULT_AUDIO = (
    "/home/smutuvi/ndizi_mlops_gemma-4"
    "/moment_3_vegetative_0bf834c6-19bb-4702-88e8-3cd2f94819e7_1713339130685.wav"
)

CHAT_PROMPTS = [
    "Habari yako?",
    "Neno 'ndizi' linamaanisha nini?",
]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_chat(model: str) -> bool:
    print("\n═══ CHAT TEST ═══")
    try:
        import litert_lm

        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        with litert_lm.Engine(model, backend=litert_lm.Backend.CPU()) as engine:
            with engine.create_conversation(messages=[
                {"role": "system", "content": [{"type": "text",
                 "text": "Wewe ni msaidizi wa lugha ya Kiswahili."}]}
            ]) as conv:
                for prompt in CHAT_PROMPTS:
                    print(f"  User : {prompt}")
                    resp = conv.send_message(prompt)
                    text = resp["content"][0]["text"]
                    print(f"  Model: {text}\n")

        print("  ✓ Chat passed")
        return True

    except Exception as e:
        print(f"  ✗ Chat failed: {e}")
        return False


def test_asr(model: str, audio: str) -> bool:
    print("\n═══ ASR TEST ═══")
    audio_path = Path(audio)
    if not audio_path.exists():
        print(f"  ✗ Audio file not found: {audio}")
        print("     Pass --audio /path/to/recording.wav to run this test.")
        return False

    try:
        import litert_lm

        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        with litert_lm.Engine(model, backend=litert_lm.Backend.CPU(), audio_backend=litert_lm.Backend.CPU()) as engine:
            with engine.create_conversation() as conv:
                resp = conv.send_message({
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": str(audio_path)},
                        {"type": "text",  "text": "Andika maneno unayosikia katika sauti hii."},
                    ],
                })
                transcript = resp["content"][0]["text"]
                print(f"  Transcript: {transcript}")

        print("  ✓ ASR passed")
        return True

    except Exception as e:
        print(f"  ✗ ASR failed: {e}")
        if "TF_LITE_AUDIO_ENCODER_HW" in str(e):
            print(
                "\n  Hint: audio encoder not in this bundle.\n"
                "  Run the slim build to get a bundle that includes it:\n"
                "    python scripts/build_litert_lm_slim.py \\\n"
                "      --skip-export \\\n"
                "      --finetuned-litertlm /path/to/existing.litertlm \\\n"
                "      --upload"
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
    ap.add_argument("--audio", default=DEFAULT_AUDIO,
                    help="Path to .wav file for ASR test")
    ap.add_argument("--no-chat", action="store_true", help="Skip Chat test")
    ap.add_argument("--no-asr",  action="store_true", help="Skip ASR test")
    args = ap.parse_args()

    model = str(Path(args.model).resolve())
    if not Path(model).exists():
        print(f"ERROR: model not found: {model}")
        return 1

    print(f"Model: {model}")
    size_gb = Path(model).stat().st_size / (1024 ** 3)
    print(f"Size : {size_gb:.2f} GB")

    chat_ok = True
    asr_ok  = True

    if not args.no_chat:
        chat_ok = test_chat(model)

    if not args.no_asr:
        asr_ok = test_asr(model, args.audio)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n═══ SUMMARY ═══")
    if not args.no_chat:
        print(f"  Chat : {'✓ passed' if chat_ok else '✗ failed'}")
    if not args.no_asr:
        print(f"  ASR  : {'✓ passed' if asr_ok  else '✗ failed'}")

    return 0 if (chat_ok and asr_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
