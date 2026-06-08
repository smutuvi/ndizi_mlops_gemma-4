"""
convert_litert.py — Convert fine-tuned Gemma-4 Swahili ASR to LiteRT (.litertlm) for
                    Android and iOS deployment.

Pipeline:
  1. Obtain merged weights (download from Hub, or merge LoRA on the fly)
  2. Convert + INT4-quantize via litert-torch export_hf → model.litertlm
  3. Validate bundle size
  4. (Optional) upload LiteRT bundle and/or merged weights to HuggingFace Hub

INT4 weights + FP32 activations (dynamic_wi4_afp32) matches the recipe used by
litert-community/gemma-4-E2B-it-litert-lm. Expected bundle: ~2 GB.

Usage:
  # Option 1 — merge LoRA on the fly, convert, upload LiteRT bundle to Hub
  python gemma-4_inference_scripts/convert_litert.py \\
      --lora-model-id smutuvi/gemma-4-e2b-sw-asr-ndizi \\
      --output-dir litert_output \\
      --upload-repo smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm \\
      --hf-token hf_...

  # Option 2 — merge LoRA on the fly, push corrected merged weights to Hub (no LiteRT)
  python gemma-4_inference_scripts/convert_litert.py \\
      --lora-model-id smutuvi/gemma-4-e2b-sw-asr-ndizi \\
      --output-dir merge_output \\
      --push-merged-to smutuvi/gemma-4-e2b-sw-asr-ndizi-merged \\
      --skip-conversion \\
      --hf-token hf_...

  # Path A — use already-merged Hub repo, then convert (fastest if merge already correct)
  python gemma-4_inference_scripts/convert_litert.py \\
      --merged-model-id smutuvi/gemma-4-e2b-sw-asr-ndizi-merged \\
      --output-dir litert_output \\
      --upload-repo smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm \\
      --hf-token hf_...

Dependencies:
  pip install litert-torch peft transformers huggingface_hub
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_MERGED_MODEL_ID = "smutuvi/gemma-4-e2b-sw-asr-ndizi-merged"
DEFAULT_BASE_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_LORA_MODEL_ID = "smutuvi/gemma-4-e2b-sw-asr-ndizi"

# Expected bundle size range (bytes) for a sanity check
MIN_BUNDLE_BYTES = 1_500_000_000   # 1.5 GB
MAX_BUNDLE_BYTES = 3_500_000_000   # 3.5 GB


# ── HF auth ───────────────────────────────────────────────────────────────────
def _setup_hf_auth(token: str | None) -> str | None:
    if not token:
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            token = os.environ.get(key, "").strip()
            if token:
                break
    if not token:
        print("WARNING: No HuggingFace token — gated/private repos will 401.")
        return None
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = token
    print(f"HF auth OK (token …{token[-4:]})")
    return token


# ── Step 1A: download already-merged model ────────────────────────────────────
def download_merged_model(model_id: str, local_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    print(f"\n[Step 1A] Downloading merged model: {model_id}")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"  Downloaded to: {path}")
    return Path(path)


# ── Step 1B: merge LoRA on the fly ────────────────────────────────────────────
def merge_lora(
    base_model_id: str,
    lora_model_id: str,
    merged_dir: Path,
    token: str | None,
) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    hub_kw = {"token": token} if token else {}
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1B] Merging LoRA into base model")
    print(f"  Base  : {base_model_id}")
    print(f"  LoRA  : {lora_model_id}")
    print(f"  Output: {merged_dir}")

    print("  Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_id, padding_side="left", **hub_kw)

    print("  Loading base model (float16)...")
    base = AutoModelForMultimodalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="sdpa",
        **hub_kw,
    )

    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base, lora_model_id, token=token)

    print("  Merging weights (merge_and_unload)...")
    model = model.merge_and_unload()

    print("  Saving merged model...")
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_dir))

    print(f"  Merged model saved to {merged_dir}")
    return merged_dir


# ── Step 2: litert-torch conversion + INT4 quantization ──────────────────────
def convert_to_litert(
    merged_dir: Path,
    output_dir: Path,
    backend: str,
) -> Path:
    try:
        from litert_torch.generative.export_hf.export import export
    except ImportError:
        raise SystemExit(
            "litert-torch is not installed.\n"
            "  pip install litert-torch"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 2] Converting to LiteRT (.litertlm)")
    print(f"  Source  : {merged_dir}")
    print(f"  Quant   : INT4 weights + FP32 activations (dynamic_wi4_afp32)")
    print(f"  Output  : {output_dir}")
    print("  This may take 30–60 minutes on CPU...")

    export(
        model=str(merged_dir),
        output_dir=str(output_dir),
        # Gemma-4 is a multimodal model — audio goes through the vision encoder
        task="image_text_to_text",
        quantization_recipe="dynamic_wi4_afp32",
        export_vision_encoder=True,
        bundle_litert_lm=True,
        keep_temporary_files=False,
        # Disable Jinja template — LiteRT runtime doesn't support dict.get() in Gemma-4 template
        use_jinja_template=False,
    )

    bundle = _find_bundle(output_dir)
    print(f"  Conversion complete: {bundle}")
    return bundle


def _find_bundle(output_dir: Path) -> Path:
    for ext in ("*.litertlm", "*.bin", "*.task"):
        matches = list(output_dir.glob(ext))
        if matches:
            return matches[0]
    raise RuntimeError(
        f"Conversion finished but no bundle file found in {output_dir}.\n"
        "Check converter logs above for errors."
    )


# ── Step 3: validate ──────────────────────────────────────────────────────────
def validate_bundle(bundle: Path) -> None:
    print(f"\n[Step 3] Validating bundle: {bundle.name}")
    size = bundle.stat().st_size
    size_gb = size / 1e9
    print(f"  Size: {size_gb:.2f} GB")
    if size < MIN_BUNDLE_BYTES:
        print(f"  WARNING: bundle is smaller than expected ({size_gb:.2f} GB < 1.5 GB). "
              "Conversion may have failed partially.")
    elif size > MAX_BUNDLE_BYTES:
        print(f"  WARNING: bundle is larger than expected ({size_gb:.2f} GB > 3.5 GB). "
              "Check quantization settings.")
    else:
        print("  Size OK.")

    # Check litert-lm CLI is available for a quick load test
    import shutil as _shutil
    if _shutil.which("litert-lm"):
        print("  litert-lm CLI found — run manually to verify:")
        print(f'    litert-lm "{bundle}" --prompt "Habari yako?"')
    else:
        print("  litert-lm CLI not found — install via: pip install litert-lm")
        print("  Then verify with:")
        print(f'    litert-lm "{bundle}" --prompt "Habari yako?"')


# ── Step 4a: upload LiteRT bundle to HuggingFace Hub (optional) ──────────────
def upload_to_hub(bundle: Path, repo_id: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    print(f"\n[Step 4a] Uploading LiteRT bundle to HuggingFace: {repo_id}")
    api = HfApi()

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    api.upload_file(
        path_or_fileobj=str(bundle),
        path_in_repo=bundle.name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add LiteRT .litertlm bundle (INT4, Swahili ASR fine-tuned)",
    )
    print(f"  Uploaded: https://huggingface.co/{repo_id}")


# ── Step 4b: push merged safetensors to HuggingFace Hub (optional) ────────────
def push_merged_to_hub(merged_dir: Path, repo_id: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    print(f"\n[Step 4b] Pushing merged model to HuggingFace: {repo_id}")
    api = HfApi()

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add correctly merged Gemma-4 Swahili ASR weights (LoRA fused into base, float16)",
    )
    print(f"  Uploaded: https://huggingface.co/{repo_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--merged-model-id",
        default=DEFAULT_MERGED_MODEL_ID,
        help=f"HF Hub ID of already-merged model (default: {DEFAULT_MERGED_MODEL_ID})",
    )
    source.add_argument(
        "--lora-model-id",
        default=None,
        help="LoRA adapter Hub ID — triggers on-the-fly merge with --base-model-id",
    )

    p.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help=f"Base model Hub ID for on-the-fly merge (default: {DEFAULT_BASE_MODEL_ID})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("litert_output"),
        help="Directory for .litertlm and intermediate files (default: ./litert_output)",
    )
    p.add_argument("--hf-token", default=None, help="HuggingFace access token")
    p.add_argument(
        "--backend", choices=("gpu", "cpu", "npu"), default="gpu",
        help="Inference backend hint (reserved for future use; litert-torch uses dynamic_wi4_afp32 regardless)",
    )
    p.add_argument(
        "--keep-merged", action="store_true",
        help="Keep local merged safetensors after conversion (default: delete to save disk)",
    )
    p.add_argument(
        "--upload-repo", default=None,
        help="HF Hub repo ID to upload the .litertlm bundle after conversion",
    )
    p.add_argument(
        "--push-merged-to", default=None,
        help="HF Hub repo ID to push the merged safetensors (use to publish/fix merged weights)",
    )
    p.add_argument(
        "--skip-conversion", action="store_true",
        help="Skip LiteRT conversion — only merge and (optionally) push merged weights",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    token = _setup_hf_auth(args.hf_token)

    merged_dir = args.output_dir / "merged"

    if args.lora_model_id:
        merged_dir = merge_lora(args.base_model_id, args.lora_model_id, merged_dir, token)
    else:
        merged_dir = download_merged_model(args.merged_model_id, merged_dir, token)

    if args.push_merged_to:
        push_merged_to_hub(merged_dir, args.push_merged_to, token)

    if args.skip_conversion:
        print("\nSkipping LiteRT conversion (--skip-conversion).")
        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
        print(f"  Merged dir : {merged_dir}")
        if args.push_merged_to:
            print(f"  Hub        : https://huggingface.co/{args.push_merged_to}")
        return

    bundle = convert_to_litert(merged_dir, args.output_dir, args.backend)

    if not args.keep_merged:
        print(f"\n  Removing intermediate merged dir: {merged_dir}")
        shutil.rmtree(merged_dir, ignore_errors=True)

    validate_bundle(bundle)

    if args.upload_repo:
        upload_to_hub(bundle, args.upload_repo, token)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Bundle : {bundle}")
    print(f"  Size   : {bundle.stat().st_size / 1e9:.2f} GB")
    if args.upload_repo:
        print(f"  Hub    : https://huggingface.co/{args.upload_repo}")
    if args.push_merged_to:
        print(f"  Merged : https://huggingface.co/{args.push_merged_to}")
    print()
    print("Next steps:")
    print("  Android — copy bundle to device and load via MediaPipe LLM Inference API")
    print("  iOS     — load via Swift LiteRT-LM API (Google AI Edge framework)")
    print(f'  CLI test: litert-lm "{bundle}" --prompt "Habari yako?"')


if __name__ == "__main__":
    main()
