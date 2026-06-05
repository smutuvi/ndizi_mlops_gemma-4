"""
eval_chat.py — Chat quality gate for Gemma-4 Swahili ASR fine-tunes.

Runs a fixed set of Swahili chat prompts against a model variant and reports a
pass/fail score. Used as a publish gate before merging or uploading a checkpoint —
a model that passes ASR WER but fails chat should be blocked from merge.

Exit codes:
  0 — pass_rate >= threshold (safe to merge)
  1 — pass_rate < threshold (chat regression detected — block merge)

Usage:
  # Test stock E2B (should pass — establishes baseline)
  python gemma-4_inference_scripts/eval_chat.py \
      --model-id google/gemma-4-E2B-it \
      --output-dir results/chat_baseline \
      --hf-token hf_...

  # Test merged Ndizi (expected to fail — quantifies regression)
  python gemma-4_inference_scripts/eval_chat.py \
      --model-id smutuvi/gemma-4-e2b-sw-asr-ndizi-merged \
      --output-dir results/chat_ndizi_merged \
      --hf-token hf_...

  # Test LoRA adapter on base
  python gemma-4_inference_scripts/eval_chat.py \
      --model-id smutuvi/gemma-4-e2b-sw-asr-ndizi \
      --base-model-id google/gemma-4-E2B-it \
      --output-dir results/chat_ndizi_adapter \
      --hf-token hf_...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_PROMPTS_FILE = SCRIPT_DIR.parent / "data" / "chat_eval_prompts.jsonl"
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_THRESHOLD = 0.80
MAX_NEW_TOKENS = 200

# Common Swahili function words — used for basic language detection.
_SWAHILI_TOKENS = {
    "na", "ya", "wa", "kwa", "ni", "la", "za", "katika", "kuwa", "au",
    "hiyo", "hii", "hizo", "hawa", "kama", "sana", "lakini", "pia",
    "pamoja", "yake", "wetu", "wake", "kwamba", "ikiwa", "je", "sijui",
    "ndiyo", "hapana", "asante", "tafadhali", "karibu", "habari", "nzuri",
    "mzuri", "sawa", "vizuri", "tumia", "pata", "fanya", "lipa", "enda",
}


# ── scoring helpers ────────────────────────────────────────────────────────────

def _is_non_empty(text: str) -> bool:
    return len(text.strip()) > 0


def _no_transcript_repetition(text: str) -> bool:
    """Reject responses that repeat the same short phrase ≥3 times (transcriber artifact)."""
    words = text.lower().split()
    if len(words) < 3:
        return True
    # Check bigrams
    for i in range(len(words) - 4):
        bigram = (words[i], words[i + 1])
        rest = " ".join(words[i + 2:])
        pattern = " ".join(bigram)
        if rest.count(pattern) >= 2:
            return False
    return True


def _not_digits_only(text: str) -> bool:
    """Reject responses that are pure numbers/digits (transcriber ASR output)."""
    return not re.fullmatch(r"[\d\s.,]+", text.strip())


def _has_swahili_content(text: str, min_fraction: float = 0.15) -> bool:
    """At least min_fraction of tokens should be recognisable Swahili words."""
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    if not tokens:
        return False
    swahili_count = sum(1 for t in tokens if t in _SWAHILI_TOKENS)
    return (swahili_count / len(tokens)) >= min_fraction


def _not_pure_asr_instruction_echo(text: str) -> bool:
    """Reject responses that echo the ASR instruction back (transcriber confusion)."""
    asr_markers = ["transcribe", "speech segment", "only output the transcription"]
    lower = text.lower()
    return not any(m in lower for m in asr_markers)


def score_response(response: str) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failed_checks)."""
    checks = [
        ("non_empty", _is_non_empty),
        ("no_repetition", _no_transcript_repetition),
        ("not_digits_only", _not_digits_only),
        ("has_swahili_content", _has_swahili_content),
        ("not_asr_echo", _not_pure_asr_instruction_echo),
    ]
    failed = [name for name, fn in checks if not fn(response)]
    return len(failed) == 0, failed


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    base_model_id: str | None,
    token: str | None,
    device_map: str,
):
    hub_kw = {"token": token} if token else {}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor_id = base_model_id or model_id
    print(f"Loading processor: {processor_id}")
    processor = AutoProcessor.from_pretrained(
        processor_id, padding_side="left", **hub_kw
    )

    print(f"Loading model: {model_id}")
    load_kw: dict = {
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
        **hub_kw,
    }
    if device_map == "auto":
        load_kw["device_map"] = "auto"
    elif torch.cuda.is_available():
        load_kw["device_map"] = {"": 0}

    if base_model_id:
        from peft import PeftModel
        base = AutoModelForMultimodalLM.from_pretrained(base_model_id, **load_kw)
        model = PeftModel.from_pretrained(base, model_id, token=token)
    else:
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kw)

    model = model.eval()
    device = next(model.parameters()).device
    print(f"Model on device: {device}")
    return model, processor, device


# ── inference ─────────────────────────────────────────────────────────────────

def run_chat_prompt(
    prompt: str,
    model,
    processor,
    device,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=inputs_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][input_len:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF Hub model ID or local path")
    p.add_argument("--base-model-id", default=None, help="Base model ID when --model-id is a LoRA adapter")
    p.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE, help="JSONL file with chat prompts")
    p.add_argument("--output-dir", type=Path, default=Path("results/chat_eval"), help="Output directory")
    p.add_argument("--hf-token", default=None, help="HuggingFace access token")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum pass_rate to exit 0 (default: 0.80)")
    p.add_argument("--device-map", choices=("cuda", "auto", "cpu"), default="cuda")
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of prompts (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    token = args.hf_token or __import__("os").environ.get("HF_TOKEN", "").strip() or None

    prompts_path = args.prompts_file
    if not prompts_path.exists():
        raise SystemExit(f"Prompts file not found: {prompts_path}")

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    if args.max_samples:
        prompts = prompts[: args.max_samples]

    model, processor, device = load_model(
        args.model_id, args.base_model_id, token, args.device_map
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    passed = 0

    for item in prompts:
        prompt_id = item.get("id", "?")
        prompt_text = item["prompt"]
        category = item.get("category", "unknown")

        print(f"  [{prompt_id}] {prompt_text[:60]}")
        response = run_chat_prompt(prompt_text, model, processor, device)
        ok, failed_checks = score_response(response)

        if ok:
            passed += 1
            status = "PASS"
        else:
            status = f"FAIL ({', '.join(failed_checks)})"

        print(f"    → {response[:80]!r}  [{status}]")
        results.append({
            "id": prompt_id,
            "category": category,
            "prompt": prompt_text,
            "response": response,
            "passed": ok,
            "failed_checks": failed_checks,
        })

    total = len(results)
    pass_rate = passed / total if total > 0 else 0.0

    metrics = {
        "model_id": args.model_id,
        "base_model_id": args.base_model_id,
        "total_prompts": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(pass_rate, 4),
        "threshold": args.threshold,
        "gate_result": "PASS" if pass_rate >= args.threshold else "FAIL",
    }

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
    for cat, counts in categories.items():
        counts["pass_rate"] = round(counts["passed"] / counts["total"], 4)
    metrics["by_category"] = categories

    out_metrics = args.output_dir / "chat_metrics.json"
    out_predictions = args.output_dir / "chat_predictions.json"

    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    out_predictions.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Chat eval: {args.model_id}")
    print(f"  Pass rate : {pass_rate:.1%}  ({passed}/{total})")
    print(f"  Threshold : {args.threshold:.1%}")
    print(f"  Gate      : {metrics['gate_result']}")
    print(f"  Results   : {out_metrics}")
    print(f"{'='*60}\n")

    sys.exit(0 if pass_rate >= args.threshold else 1)


if __name__ == "__main__":
    main()
