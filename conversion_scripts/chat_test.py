"""
chat_test.py
─────────────
Test Swahili chat capability of a LoRA checkpoint.
Loads the finetuned adapter and runs text-only chat prompts
(no audio) to verify conversational ability after ASR finetuning.

Usage:
  python conversion_scripts/chat_test.py
  python conversion_scripts/chat_test.py --checkpoint runs/my-run/best
  python conversion_scripts/chat_test.py --prompt "Unaishi wapi?"
  python conversion_scripts/chat_test.py --no-adapter   # test base model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best"

# Default prompts to test general Swahili chat
DEFAULT_PROMPTS = [
    "Habari yako?",
    "Nchi ya Kenya iko wapi Afrika?",
    "Nipe muhtasari mfupi wa historia ya Kiswahili.",
]


def chat_once(model, processor, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    # Cast fp inputs to model dtype
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(dtype=model.dtype)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    new_tokens = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="LoRA adapter directory (default: artifacts/checkpoints/best)")
    ap.add_argument("--no-adapter", action="store_true",
                    help="Load base model without LoRA adapter (for baseline comparison)")
    ap.add_argument("--prompt", default=None,
                    help="Single chat prompt (skips default prompts)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.no_adapter:
        from src.eval.baseline import load_baseline_gemma
        model, processor = load_baseline_gemma(fp16=False)
        print("Model     : base (no LoRA adapter)")
    else:
        from src.eval.finetuned import load_finetuned_gemma
        checkpoint = Path(args.checkpoint).expanduser()
        print(f"Checkpoint: {checkpoint.resolve()}")
        model, processor, _ = load_finetuned_gemma(checkpoint)

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print()
    for i, prompt in enumerate(prompts, 1):
        print(f"{'─' * 60}")
        print(f"[{i}] USER : {prompt}")
        reply = chat_once(model, processor, prompt, max_new_tokens=args.max_new_tokens)
        print(f"    BOT  : {reply}")

    print(f"{'─' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
