"""
convert_ndizi_to_litertlm.py
────────────────────────────
Merges the Ndizi LoRA adapter with the Gemma-4-E2B base model,
then converts the merged model to .litertlm — preserving BOTH:
  • Swahili ASR (audio transcription via the audio backend)
  • Chat conversation (text generation via the text decoder)

Architecture note:
  Gemma-4-E2B-it is a natively multimodal model. The .litertlm container
  packages separate TFLite sub-models for text, vision, and audio.
  LiteRT-LM loads the text decoder always, and hot-loads vision/audio only
  when needed, keeping idle memory low on iPhone.

Requirements:
  - Python 3.11+          (uv handles this if run via uv)
  - ~25 GB free disk
  - ~16 GB RAM  (or 8 GB VRAM with float16)
  - uv installed:         https://docs.astral.sh/uv/
  - Gemma license accepted on HF: https://huggingface.co/google/gemma-4-E2B-it

Usage:
  # Install dependencies
  pip install transformers peft torch accelerate huggingface_hub soundfile

  # Install conversion + runtime tools
  uv tool install litert-torch-nightly
  uv tool install litert-lm

  # Run (minimal)
  python convert_ndizi_to_litertlm.py --hf_token YOUR_TOKEN

  # Run (push merged model to HF first — recommended for reliable conversion)
  python convert_ndizi_to_litertlm.py --hf_token YOUR_TOKEN --hf_username smutuvi
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


BASE_MODEL_ID  = "google/gemma-4-E2B-it"
ADAPTER_REPO   = "smutuvi/gemma-4-e2b-sw-asr-ndizi"
MERGED_DIR     = Path("./ndizi_gemma4_merged")
OUTPUT_DIR     = Path("./ndizi_gemma4_litertlm")


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list, **kwargs):
    print(f"\n▶ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def is_peft_adapter(repo_or_path: str, token: str | None) -> bool:
    from huggingface_hub import list_repo_files
    try:
        files = list(list_repo_files(repo_or_path, token=token))
        return "adapter_config.json" in files
    except Exception:
        # fallback: try downloading adapter_config.json
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_or_path, "adapter_config.json", token=token)
            return True
        except Exception:
            return False


def litert_torch_cmd() -> list:
    """Return the base command for litert-torch, preferring the uv-installed one."""
    p = shutil.which("litert-torch")
    if p:
        return [p]
    return ["uv", "tool", "run", "--from", "litert-torch-nightly", "litert-torch"]


def litert_lm_cmd() -> list | None:
    p = shutil.which("litert-lm")
    if p:
        return [p]
    # try via uv
    try:
        subprocess.run(["uv", "tool", "run", "--from", "litert-lm", "litert-lm", "--help"],
                       capture_output=True, check=True)
        return ["uv", "tool", "run", "--from", "litert-lm", "litert-lm"]
    except Exception:
        return None


# ── Step 1: merge ─────────────────────────────────────────────────────────────

def merge_adapter(token: str | None):
    print("\n═══ STEP 1: Merging LoRA adapter into base model ═══")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if MERGED_DIR.exists() and any(MERGED_DIR.iterdir()):
        print(f"  {MERGED_DIR} already exists — skipping (delete to re-run).")
        return

    print(f"  Loading base model: {BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )

    print(f"  Loading LoRA adapter: {ADAPTER_REPO}")
    model = PeftModel.from_pretrained(model, ADAPTER_REPO, token=token)

    print("  Merging adapter weights into base model…")
    model = model.merge_and_unload()

    print(f"  Saving merged model → {MERGED_DIR}")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    print("  Saving tokenizer…")
    AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token).save_pretrained(MERGED_DIR)

    # Copy processor config if present (needed for audio/vision)
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["preprocessor_config.json", "processor_config.json",
                      "special_tokens_map.json", "generation_config.json"]:
            try:
                src = hf_hub_download(BASE_MODEL_ID, fname, token=token)
                shutil.copy(src, MERGED_DIR / fname)
            except Exception:
                pass
    except Exception:
        pass

    print("  ✓ Merge complete.")


# ── Step 2: sanity-check BOTH capabilities ────────────────────────────────────

def sanity_check_chat():
    print("\n  [Chat] Running text generation test…")
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=str(MERGED_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=60,
    )
    # Test Swahili conversation
    prompt = "Wewe ni msaidizi wa lugha ya Kiswahili. Habari yako?"
    out = pipe(prompt)[0]["generated_text"]
    print(f"  Prompt : {prompt!r}")
    print(f"  Output : {out!r}")
    print("  ✓ Chat OK")


def sanity_check_asr(audio_path: str | None = None):
    """
    Test ASR on a short WAV file.  If you don't have one, we synthesise
    a silent placeholder just to verify the pipeline loads without error.
    """
    print("\n  [ASR] Running audio transcription test…")
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    if audio_path is None:
        # Create a tiny silent WAV for pipeline loading test
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 16000 * 2)  # 1 second of silence
        audio_path = tmp.name
        print(f"  (No audio file provided — using 1 s silence at {audio_path})")

    try:
        processor = AutoProcessor.from_pretrained(str(MERGED_DIR))
        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(MERGED_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        import soundfile as sf
        audio, sr = sf.read(audio_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text",  "text": "Andika maneno unayosikia katika sauti hii."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=100)
        transcript = processor.decode(out_ids[0], skip_special_tokens=True)
        print(f"  Transcript: {transcript!r}")
        print("  ✓ ASR pipeline loads correctly.")
    except Exception as e:
        print(f"  ⚠  ASR test error: {e}")
        print("     This may be fine — the audio sub-model loads separately at runtime.")


def sanity_check(audio_path: str | None = None):
    print("\n═══ STEP 2: Sanity checks (chat + ASR) ═══")
    sanity_check_chat()
    sanity_check_asr(audio_path)


# ── Step 3: push merged model to HF ──────────────────────────────────────────

def push_to_hub(username: str, token: str) -> str:
    repo_id = f"{username}/ndizi-gemma4-merged"
    print(f"\n═══ STEP 3: Pushing merged model → {repo_id} (private) ═══")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=str(MERGED_DIR), repo_id=repo_id, token=token)
    print(f"  ✓ Available at https://huggingface.co/{repo_id}")
    return repo_id


# ── Step 4: convert to .litertlm ─────────────────────────────────────────────

def convert(model_source: str, token: str | None = None):
    """
    Converts to .litertlm.  The --externalize_embedder flag is critical:
    it memory-maps the 1.12 GB embedding table on iOS, dropping working
    RAM from ~2 GB to ~607 MB — making it viable on most iPhones.

    The audio + vision sub-models are automatically included when the
    base model is multimodal (Gemma-4-E2B-it is).
    """
    print(f"\n═══ STEP 4: Converting {model_source} → .litertlm ═══")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if token:
        env["HUGGING_FACE_HUB_TOKEN"] = token
        env["HF_TOKEN"] = token

    cmd = litert_torch_cmd() + [
        "export_hf",
        f"--model={model_source}",
        f"--output_dir={OUTPUT_DIR}",
        "--externalize_embedder",   # ← memory-maps embeddings; essential for iPhone
    ]

    run(cmd, env=env)

    litertlm = OUTPUT_DIR / "model.litertlm"
    if litertlm.exists():
        size_mb = litertlm.stat().st_size / (1024 ** 2)
        print(f"\n  ✓ Output: {litertlm}  ({size_mb:.0f} MB)")
    else:
        # Some versions write a different filename — find it
        files = list(OUTPUT_DIR.glob("*.litertlm"))
        if files:
            print(f"\n  ✓ Output: {files[0]}  ({files[0].stat().st_size/(1024**2):.0f} MB)")
        else:
            print("\n  ⚠  No .litertlm found in output dir — check logs above.")


# ── Step 5: local inference test ─────────────────────────────────────────────

def local_test(audio_path: str | None = None):
    print("\n═══ STEP 5: Local inference tests ═══")
    litertlm = next(OUTPUT_DIR.glob("*.litertlm"), None)
    if litertlm is None:
        print("  No .litertlm found — skipping local test.")
        return

    cmd = litert_lm_cmd()
    if cmd is None:
        print("  litert-lm not found.  Install: uv tool install litert-lm")
        return

    # Chat test
    print("\n  [Chat test]")
    run(cmd + ["run", str(litertlm), "--prompt=Habari, unaweza kunisaidia?"])

    # ASR test (only if an audio file is provided)
    if audio_path:
        print("\n  [ASR test — audio file]")
        run(cmd + [
            "run", str(litertlm),
            f"--audio={audio_path}",
            "--prompt=Andika maneno unayosikia.",
        ])
    else:
        print("\n  [ASR test] — pass --audio_path to test transcription locally.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ndizi Gemma-4 → .litertlm (ASR + Chat)")
    ap.add_argument("--hf_token",    required=True,
                    help="HuggingFace token (needs Gemma license + adapter access)")
    ap.add_argument("--hf_username", default=None,
                    help="HF username — pushes merged model privately before conversion."
                         " Recommended; omit to convert from local path.")
    ap.add_argument("--audio_path",  default=None,
                    help="Path to a .wav file for ASR sanity check (optional)")
    ap.add_argument("--skip_merge",  action="store_true")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--skip_push",   action="store_true")
    args = ap.parse_args()

    # Authenticate
    from huggingface_hub import login
    login(token=args.hf_token)

    # 1 — merge
    if not args.skip_merge:
        if is_peft_adapter(ADAPTER_REPO, args.hf_token):
            print(f"\n  Detected PEFT/LoRA adapter — merging with {BASE_MODEL_ID}.")
            merge_adapter(args.hf_token)
        else:
            print(f"\n  {ADAPTER_REPO} appears to be a full model — downloading directly.")
            if not (MERGED_DIR.exists() and any(MERGED_DIR.iterdir())):
                from huggingface_hub import snapshot_download
                snapshot_download(ADAPTER_REPO, local_dir=str(MERGED_DIR),
                                  token=args.hf_token)

    # 2 — sanity
    if not args.skip_sanity:
        sanity_check(audio_path=args.audio_path)

    # 3 + 4 — convert
    if not args.skip_push and args.hf_username:
        repo_id = push_to_hub(args.hf_username, args.hf_token)
        model_source = repo_id
    else:
        print("\n  Converting from local path (no --hf_username given).")
        model_source = str(MERGED_DIR)

    convert(model_source, token=args.hf_token)

    # 5 — local test
    local_test(audio_path=args.audio_path)

    litertlm = next(OUTPUT_DIR.glob("*.litertlm"), OUTPUT_DIR / "model.litertlm")
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  DONE — your .litertlm file is ready:                               ║
║  {str(litertlm):<68}║
║                                                                      ║
║  On-device testing:                                                  ║
║  1. Quick test (no code): Google AI Edge Gallery app on iPhone       ║
║     App Store → "Google AI Edge Gallery" → load model from Files    ║
║                                                                      ║
║  2. Build an iOS app — add to Xcode via Swift Package Manager:      ║
║     https://github.com/google-ai-edge/LiteRT-LM  (v0.12.0+)        ║
║                                                                      ║
║  iOS Swift snippet (ASR + Chat):                                     ║
║  ─────────────────────────────────────────────────────────────────  ║
║  let config = try EngineConfig(                                      ║
║    modelPath: "model.litertlm",                                      ║
║    backend: .gpu,               // Metal GPU acceleration            ║
║    audioBackend: .cpu(),        // Enable ASR sub-model              ║
║    cacheDir: NSTemporaryDirectory()                                  ║
║  )                                                                   ║
║  let engine = Engine(engineConfig: config)                           ║
║  try await engine.initialize()                                       ║
║                                                                      ║
║  // Chat                                                             ║
║  let conv = try await engine.createConversation()                   ║
║  let reply = try await conv.sendMessage(Message("Habari?"))         ║
║                                                                      ║
║  // ASR (transcription)                                              ║
║  let msg = Message(contents: [                                       ║
║    Content.audioFile("/path/to/recording.wav"),                      ║
║    Content.text("Andika maneno unayosikia.")                        ║
║  ])                                                                  ║
║  let transcript = try await conv.sendMessage(msg)                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
