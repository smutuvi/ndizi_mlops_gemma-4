"""
fractional_merge_test.py
────────────────────────
Merge a LoRA adapter at a fractional scale (0 < scale ≤ 1.0) and run
combined ASR + Chat tests. Use this to find the sweet spot between ASR
performance and chat quality without retraining.

How it works:
  The LoRA contribution is  W += (alpha/r) * B @ A
  Scaling B by `scale` gives W += scale * (alpha/r) * B @ A
  scale=1.0 → full adapter (best ASR, possible chat degradation)
  scale=0.0 → base model  (no ASR gain, perfect chat)
  scale=0.6-0.8 → typical sweet spot

Usage:
  python conversion_scripts/fractional_merge_test.py --scale 0.7
  python conversion_scripts/fractional_merge_test.py --scale 0.7 --scale 0.5 --scale 0.9
  python conversion_scripts/fractional_merge_test.py \\
    --checkpoint runs/gemma4-e2b-simple-instruction-2026-06-13/best \\
    --scale 0.7 \\
    --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best"
DEFAULT_DATASETS = ["smutuvi/ndizi-1:test", "smutuvi/ndizi-1-2025:test"]
CHUNK_S = 30.0

DEFAULT_CHAT_PROMPTS = [
    "Habari yako?",
    "Nchi ya Kenya iko wapi Afrika?",
    "Nipe muhtasari mfupi wa historia ya Kiswahili.",
]


def apply_fractional_scale(model, scale: float) -> None:
    """Scale all LoRA B matrices in-place by `scale`."""
    import torch
    scaled = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_B"):
            for key, layer in module.lora_B.items():
                layer.weight.data = layer.weight.data * scale
                scaled += 1
    print(f"  Scaled {scaled} lora_B matrices by {scale:.3f}")


def load_and_scale(checkpoint: Path, scale: float):
    """Load finetuned model, scale the adapter, then merge and unload."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from src.utils.runtime import get_runtime

    rt = get_runtime()
    print(f"  Loading base model: {rt.base_model_id}")
    processor = AutoProcessor.from_pretrained(rt.base_model_id, padding_side="left")
    base = AutoModelForMultimodalLM.from_pretrained(
        rt.base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"  Attaching adapter: {checkpoint}")
    peft_model = PeftModel.from_pretrained(base, str(checkpoint))

    if scale < 1.0:
        apply_fractional_scale(peft_model, scale)
    else:
        print("  scale=1.0 — no rescaling applied")

    print("  Merging and unloading adapter...")
    model = peft_model.merge_and_unload()
    model.eval()
    return model, processor


def run_asr(model, processor, datasets, n):
    import datasets as hf_datasets
    from src.inference.chunked_transcribe import gemma_transcribe_chunked
    from src.utils.constants import ASR_INSTRUCTION

    def transcribe(audios):
        return gemma_transcribe_chunked(
            model, processor, audios,
            chunk_length_s=CHUNK_S,
            instruction=ASR_INSTRUCTION,
        )

    print(f"\n  {'─'*54}")
    print("  ASR SAMPLES")
    print(f"  {'─'*54}")
    for spec in datasets:
        repo, _, split = spec.partition(":")
        split = split or "test"
        print(f"\n  Dataset: {repo}  split={split}  n={n}")
        ds = hf_datasets.load_dataset(repo, split=f"{split}[:{n}]")
        for i, row in enumerate(ds):
            hyp = transcribe([row["audio"]])[0]
            ref = row.get("text", row.get("sentence", ""))
            print(f"    [{i+1}] REF: {ref}")
            print(f"         HYP: {hyp}")


def run_chat(model, processor, prompts, max_new_tokens):
    import torch

    print(f"\n  {'─'*54}")
    print("  CHAT SAMPLES")
    print(f"  {'─'*54}")
    for i, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inputs[k] = v.to(dtype=model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = processor.batch_decode(
            out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )[0].strip()
        print(f"\n  [{i}] USER: {prompt}")
        print(f"       BOT : {reply}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="LoRA adapter directory")
    ap.add_argument("--scale", type=float, nargs="+", default=[0.7],
                    help="Merge scale(s) to test (default: 0.7). Pass multiple to compare.")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--n", type=int, default=3, help="ASR samples per dataset")
    ap.add_argument("--chat-prompts", nargs="+", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--save",
        default=None,
        metavar="DIR",
        help="Save the merged model to DIR (only valid when a single --scale is given). "
             "Use the saved path as --merged-model for build_litert_lm_slim.py.",
    )
    args = ap.parse_args()

    if args.save and len(args.scale) > 1:
        ap.error("--save can only be used with a single --scale value")

    checkpoint = Path(args.checkpoint).expanduser()
    prompts = args.chat_prompts or DEFAULT_CHAT_PROMPTS

    for scale in sorted(set(args.scale), reverse=True):
        print(f"\n{'═'*60}")
        print(f"  SCALE = {scale:.2f}  |  checkpoint: {checkpoint.name}")
        print(f"{'═'*60}")

        model, processor = load_and_scale(checkpoint, scale)

        if args.save:
            save_dir = Path(args.save)
            print(f"\n  Saving merged model to {save_dir} ...")
            model.save_pretrained(str(save_dir))
            processor.save_pretrained(str(save_dir))
            print(f"  Saved. Use with:")
            print(f"    python scripts/build_litert_lm_slim.py --merged-model {save_dir}")

        run_asr(model, processor, args.datasets, args.n)
        run_chat(model, processor, prompts, args.max_new_tokens)

        print(f"\n{'═'*60}")

        # Free memory between runs
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
