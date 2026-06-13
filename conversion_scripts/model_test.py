"""
model_test.py
─────────────
Combined ASR + Chat sanity-check for a LoRA checkpoint.

Usage:
  python conversion_scripts/model_test.py
  python conversion_scripts/model_test.py --checkpoint runs/my-run/best
  python conversion_scripts/model_test.py --mode asr
  python conversion_scripts/model_test.py --mode chat
  python conversion_scripts/model_test.py --audio /path/to/file.wav
  python conversion_scripts/model_test.py --no-adapter        # base model baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best"
DEFAULT_DATASETS = ["smutuvi/ndizi-1:test", "smutuvi/ndizi-1-2025:test"]
DEFAULT_N = 3
CHUNK_S = 30.0

DEFAULT_CHAT_PROMPTS = [
    "Habari yako?",
    "Nchi ya Kenya iko wapi Afrika?",
    "Nipe muhtasari mfupi wa historia ya Kiswahili.",
]


# ── Chat helpers ──────────────────────────────────────────────────────────────

def chat_once(model, processor, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

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

    new_tokens = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def run_chat_tests(model, processor, prompts, max_new_tokens=256):
    print(f"\n{'═' * 60}")
    print("CHAT TEST")
    print(f"{'═' * 60}")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  [{i}] USER : {prompt}")
        reply = chat_once(model, processor, prompt, max_new_tokens=max_new_tokens)
        print(f"       BOT  : {reply}")


# ── ASR helpers ───────────────────────────────────────────────────────────────

def run_asr_tests(model, processor, args):
    import datasets as hf_datasets
    from src.inference.chunked_transcribe import gemma_transcribe_chunked
    from src.inference.gemma_inputs import load_audio_file
    from src.utils.constants import ASR_INSTRUCTION, PUNCTUATION_ASR_INSTRUCTION

    instruction = (
        PUNCTUATION_ASR_INSTRUCTION if args.instruction == "punctuation" else ASR_INSTRUCTION
    )

    def transcribe(audios):
        return gemma_transcribe_chunked(
            model, processor, audios,
            chunk_length_s=CHUNK_S,
            instruction=instruction,
        )

    print(f"\n{'═' * 60}")
    print("ASR TEST")
    print(f"{'═' * 60}")

    # Single file mode
    if args.audio:
        audio = load_audio_file(args.audio)
        hyp = transcribe([audio])[0]
        print(f"\n  File : {Path(args.audio).name}")
        print(f"  HYP  : {hyp}")
        return

    # Dataset sampling mode
    for spec in args.datasets:
        repo, _, split = spec.partition(":")
        split = split or "test"
        print(f"\n  Dataset : {repo}  split={split}  n={args.n}")
        print(f"  {'─' * 50}")
        ds = hf_datasets.load_dataset(repo, split=f"{split}[:{args.n}]")
        for i, row in enumerate(ds):
            hyp = transcribe([row["audio"]])[0]
            ref = row.get("text", row.get("sentence", ""))
            print(f"\n  Sample {i + 1}")
            print(f"    REF : {ref}")
            print(f"    HYP : {hyp}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="LoRA adapter directory (default: artifacts/checkpoints/best)")
    ap.add_argument("--model", default=None,
                    help="Override base model (e.g. smutuvi/gemma-4-e2b-sw-asr-ndizi-merged). "
                         "Use with --no-adapter to test a merged Hub model.")
    ap.add_argument("--no-adapter", action="store_true",
                    help="Load model directly without LoRA adapter (use with --model for merged Hub models)")
    ap.add_argument("--mode", choices=("asr", "chat", "both"), default="both",
                    help="Which tests to run (default: both)")

    # ASR args
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="HF dataset splits to sample from (format: repo:split)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="Samples per dataset (default: 3)")
    ap.add_argument("--audio", default=None,
                    help="Single audio file path (skips dataset sampling)")
    ap.add_argument("--instruction", choices=("default", "punctuation"), default="default")

    # Chat args
    ap.add_argument("--chat-prompts", nargs="+", default=None,
                    help="Custom chat prompts (default: 3 built-in Swahili prompts)")
    ap.add_argument("--max-new-tokens", type=int, default=256)

    args = ap.parse_args()

    # Apply model override if given (e.g. merged Hub model)
    if args.model:
        from src.utils.runtime import apply_model_choice
        apply_model_choice(args.model)

    # Load model
    if args.no_adapter:
        from src.eval.baseline import load_baseline_gemma
        model, processor = load_baseline_gemma(fp16=False)
        label = args.model or "base (no LoRA adapter)"
        print(f"Model     : {label}")
    else:
        from src.eval.finetuned import load_finetuned_gemma
        checkpoint = Path(args.checkpoint).expanduser()
        print(f"Checkpoint: {checkpoint.resolve()}")
        model, processor, _ = load_finetuned_gemma(checkpoint)

    # Run tests
    if args.mode in ("asr", "both"):
        run_asr_tests(model, processor, args)

    if args.mode in ("chat", "both"):
        prompts = args.chat_prompts or DEFAULT_CHAT_PROMPTS
        run_chat_tests(model, processor, prompts, max_new_tokens=args.max_new_tokens)

    print(f"\n{'═' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
