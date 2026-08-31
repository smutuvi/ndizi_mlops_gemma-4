#!/usr/bin/env python3
"""
AWQ calibrated INT4 quantization for Sunbird/Sunflower-Gemma4-E2B.

WHY THIS HELPS
--------------
LiteRT's `dynamic_wi4_afp32` does un-calibrated per-channel INT4.  It picks
scale factors that minimise the per-tensor error but ignores which weights are
"activation-sensitive".  AWQ (Activation-aware Weight Quantization) analyses
a small calibration set, identifies the ≈1% of channels that matter most, and
protects them — giving INT4 quality much closer to INT8 at the same file size.

After AWQ the model is saved in standard HF safetensors format and can be
re-exported to LiteRT exactly like the original fp16 model.

REQUIREMENTS
------------
  pip install autoawq transformers accelerate huggingface_hub

  GPU strongly recommended — on a single A100 (40 GB) quantization takes
  ~20 min.  CPU-only will work but may take 2–4 hours for a 4-B model.

USAGE
-----
  # Quantize (outputs to artifacts/sunflower-gemma4-e2b-awq/)
  python scripts/awq_quantize.py

  # Then export to LiteRT from the AWQ model:
  conda run -n ndizi python scripts/build_litert_lm_slim.py \\
      --merged-model artifacts/sunflower-gemma4-e2b-awq \\
      --quantization dynamic_wi4_afp32 \\
      --jinja-template-override sunbird \\
      --output-name sunflower-gemma4-e2b-awq-litert.litertlm

NOTE on double-quantization
---------------------------
`dynamic_wi4_afp32` will re-quantize the AWQ INT4 weights through a
dequant→requant cycle.  This sounds wasteful but in practice the AWQ
weight layout is already highly quantization-friendly, so LiteRT's dynamic
INT4 step loses very little additional accuracy.  Alternatively pass
`--quantization dynamic_wi8_afp32` to get a slightly larger (~3–3.5 GB)
bundle with zero re-quantization loss from the AWQ step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "Sunbird/Sunflower-Gemma4-E2B"
DEFAULT_CALIB      = str(ROOT / "calibration_data" / "swahili_calib.jsonl")
DEFAULT_OUTPUT     = str(ROOT / "artifacts" / "sunflower-gemma4-e2b-awq")
DEFAULT_N_SAMPLES  = 128
DEFAULT_SEQLEN     = 512   # tokens per calibration sample
DEFAULT_GROUP_SIZE = 128   # AWQ weight group size (128 = standard)
DEFAULT_SYSTEM     = "Wewe ni msaidizi wa lugha ya Kiswahili."


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help=f"HF model id or local path (default: {DEFAULT_MODEL})")
    p.add_argument("--calib",      default=DEFAULT_CALIB,
                   help=f"JSONL calibration file (default: {DEFAULT_CALIB})")
    p.add_argument("--output",     default=DEFAULT_OUTPUT,
                   help=f"Output directory for AWQ model (default: {DEFAULT_OUTPUT})")
    p.add_argument("--n-samples",  type=int, default=DEFAULT_N_SAMPLES,
                   help=f"Calibration samples to use (default: {DEFAULT_N_SAMPLES})")
    p.add_argument("--seqlen",     type=int, default=DEFAULT_SEQLEN,
                   help=f"Max tokens per sample (default: {DEFAULT_SEQLEN})")
    p.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                   help=f"AWQ weight group size (default: {DEFAULT_GROUP_SIZE})")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM,
                   help=f"System prompt for chat formatting (default: '{DEFAULT_SYSTEM}')")
    p.add_argument("--hf-token",   default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--upload",     default=None, metavar="HF_REPO",
                   help="Upload AWQ model to this HF repo after quantization")
    args = p.parse_args()

    # ── 1. Load tokenizer ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Model  : {args.model}")
    print(f"Calib  : {args.calib}")
    print(f"Output : {args.output}")
    print(f"{'─'*60}\n")

    try:
        from transformers import AutoTokenizer
        from awq import AutoAWQForCausalLM
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nInstall with:  pip install autoawq transformers accelerate")
        return 1

    print("Loading tokenizer...")
    tok_kwargs = {}
    if args.hf_token:
        tok_kwargs["token"] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)

    # ── 2. Build calibration texts ─────────────────────────────────────────────
    print(f"Loading calibration data from {args.calib} ...")
    calib_path = Path(args.calib)
    if not calib_path.exists():
        print(f"ERROR: calibration file not found: {calib_path}")
        print("  Run: python scripts/awq_quantize.py --calib <path>")
        return 1

    raw: list[dict] = []
    with calib_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    raw = raw[: args.n_samples]
    print(f"  Loaded {len(raw)} samples")

    calib_texts: list[str] = []
    for sample in raw:
        system = sample.get("system", args.system_prompt)
        messages: list[dict] = [{"role": "system", "content": system}]
        if "user" in sample:
            messages.append({"role": "user", "content": sample["user"]})
        if "assistant" in sample:
            messages.append({"role": "assistant", "content": sample["assistant"]})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: plain concatenation using Sunbird token format
            text = (
                f"<|turn>system\n{system}<turn|>\n"
                + (f"<|turn>user\n{sample['user']}<turn|>\n" if "user" in sample else "")
                + (f"<|turn>model\n{sample['assistant']}<turn|>\n" if "assistant" in sample else "")
            )
        calib_texts.append(text)

    # Show a preview of one sample
    print(f"\n  Sample [0] preview ({len(calib_texts[0])} chars):")
    print("  " + calib_texts[0][:200].replace("\n", "\\n") + "...")

    # ── 3. Load model and quantize ─────────────────────────────────────────────
    print(f"\nLoading model {args.model} (may take a few minutes)...")
    model_kwargs = {"safetensors": True}
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    model = AutoAWQForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        **model_kwargs,
    )

    quant_config = {
        "zero_point": True,          # zero-point quantization (slightly better quality)
        "q_group_size": args.group_size,
        "w_bit": 4,                  # INT4 weights
        "version": "GEMM",           # GEMM kernel (fast on GPU; CPU fallback works too)
    }
    print(f"\nAWQ quantization config: {quant_config}")
    print("Starting quantization (this will take 15-60 minutes depending on hardware)...")

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_seq_len=args.seqlen,
    )

    # ── 4. Save ────────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving AWQ model to {output_dir} ...")
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("  ✓ Saved")

    # Quick size check
    total_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e6
    print(f"  Model size on disk: {total_mb:.0f} MB")

    # ── 5. Optional upload ─────────────────────────────────────────────────────
    if args.upload:
        print(f"\nUploading to HF repo: {args.upload} ...")
        from huggingface_hub import HfApi
        api = HfApi(token=args.hf_token)
        api.create_repo(args.upload, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.upload,
            commit_message="AWQ INT4 calibrated quantization (Swahili calibration data)",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{args.upload}")

    # ── 6. Next step instructions ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("DONE — AWQ model ready")
    print(f"{'═'*60}")
    print(f"\nNext: export to LiteRT bundle\n")
    print(f"  Option A — LiteRT INT4 (smallest, ~2.5 GB, best shot at good quality):")
    print(f"    conda run -n ndizi python scripts/build_litert_lm_slim.py \\")
    print(f"      --merged-model {output_dir} \\")
    print(f"      --quantization dynamic_wi4_afp32 \\")
    print(f"      --jinja-template-override sunbird \\")
    print(f"      --output-name sunflower-awq-int4.litertlm")
    print()
    print(f"  Option B — LiteRT INT8 from AWQ model (~3.5 GB, safest quality):")
    print(f"    conda run -n ndizi python scripts/build_litert_lm_slim.py \\")
    print(f"      --merged-model {output_dir} \\")
    print(f"      --quantization dynamic_wi8_afp32 \\")
    print(f"      --jinja-template-override sunbird \\")
    print(f"      --output-name sunflower-awq-int8.litertlm")
    print()
    print("  Then test:")
    print(f"    conda run -n ndizi python conversion_scripts/quick_test.py \\")
    print(f"      --model artifacts/litert_slim/sunflower-awq-int4.litertlm --no-asr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
