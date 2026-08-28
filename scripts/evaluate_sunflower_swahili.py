#!/usr/bin/env python3
"""Evaluate Sunflower-Gemma4-E2B vs an Ndizi LoRA checkpoint — ASR + Swahili chat.

ASR eval sets are held-out: Ndizi test (in-domain) and FLEURS sw_ke test (OOD).
By default both the **base model** and the **LoRA checkpoint** are scored so you
can see whether Ndizi WER improved.

  python scripts/evaluate_sunflower_swahili.py \\
    --checkpoint artifacts/checkpoints_sunflower_ndizi/best \\
    --output-dir eval/sunflower-ndizi \\
    --max-samples 32

  # Base only
  python scripts/evaluate_sunflower_swahili.py --no-adapter --max-samples 32

  # Checkpoint only (skip base)
  python scripts/evaluate_sunflower_swahili.py --skip-base --max-samples 32
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _free_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _chat_once(model, processor, prompt: str, system_prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]
    inputs = processor(text=text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "is_floating_point") and v.is_floating_point():
            inputs[k] = v.to(dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0].strip()


def _chat_ok(prompt: str, reply: str) -> tuple[bool, str]:
    t = (reply or "").strip()
    if not t:
        return False, "empty"
    if t.casefold() == prompt.strip().casefold():
        return False, "echo"
    if len(t.split()) < 3:
        return False, "too_short"
    return True, "ok"


def load_eval_model(args):
    from src.eval.baseline import load_baseline_gemma
    from src.eval.finetuned import load_finetuned_gemma

    if args.no_adapter:
        model, processor = load_baseline_gemma(fp16=args.fp16)
        return model, processor
    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(
            f"LoRA checkpoint not found: {ckpt}\n"
            "Train first, or pass --no-adapter to score the base Sunflower weights."
        )
    model, processor, _ = load_finetuned_gemma(ckpt, fp16=args.fp16)
    return model, processor


def run_chat_eval(args, model, processor, out_dir: Path) -> dict:
    prompts_path = Path(args.chat_prompts)
    prompts: list[dict] = []
    if prompts_path.is_file():
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                if isinstance(rec, str):
                    rec = {"prompt": rec}
                prompts.append(rec)
    for extra in args.chat_prompt_list or []:
        prompts.append({"prompt": extra})

    rows = []
    n_pass = 0
    print(f"\n{'═'*60}\nCHAT EVAL  system={args.system_prompt!r}\n{'═'*60}")
    for i, rec in enumerate(prompts, 1):
        prompt = rec.get("prompt") or rec.get("user") or ""
        reply = _chat_once(model, processor, prompt, args.system_prompt, args.max_new_tokens)
        ok, reason = _chat_ok(prompt, reply)
        n_pass += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  [{i}] {mark} {prompt}\n       {reply[:240]}")
        rows.append({"prompt": prompt, "reply": reply, "ok": ok, "reason": reason})

    n = max(len(rows), 1)
    summary = {"n": len(rows), "n_pass": n_pass, "pass_rate": n_pass / n, "rows": rows}
    (out_dir / "chat.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Chat pass_rate={summary['pass_rate']:.2f}  ({n_pass}/{len(rows)})")
    print("  Wrote", out_dir / "chat.json")
    return summary


def _is_ndizi(name: str) -> bool:
    return "ndizi" in name.lower()


def _weighted_wer(per_set: dict, predicate) -> dict | None:
    rows = [v for k, v in per_set.items() if predicate(k) and v.get("wer") is not None]
    n = sum(int(r.get("n") or 0) for r in rows)
    if not n:
        return None
    wer = sum(float(r["wer"]) * int(r.get("n") or 0) for r in rows) / n
    cer_rows = [r for r in rows if r.get("cer") is not None]
    cer = None
    if cer_rows:
        cn = sum(int(r.get("n") or 0) for r in cer_rows)
        cer = sum(float(r["cer"]) * int(r.get("n") or 0) for r in cer_rows) / max(cn, 1)
    return {"wer": wer, "cer": cer, "n": n}


def _fmt(x) -> str:
    if x is None:
        return "   n/a"
    return f"{float(x):6.3f}"


def compare_asr(base_metrics: dict | None, tuned_metrics: dict | None) -> dict:
    base_sets = (base_metrics or {}).get("per_set") or {}
    tuned_sets = (tuned_metrics or {}).get("per_set") or {}
    names = list(dict.fromkeys([*base_sets, *tuned_sets]))
    rows = []
    for name in names:
        b = base_sets.get(name) or {}
        t = tuned_sets.get(name) or {}
        bw, tw = b.get("wer"), t.get("wer")
        delta = None if bw is None or tw is None else float(tw) - float(bw)
        improved = delta is not None and delta < -1e-6
        rows.append(
            {
                "set": name,
                "ndizi": _is_ndizi(name),
                "n": t.get("n") or b.get("n"),
                "wer_base": bw,
                "wer_lora": tw,
                "delta_wer": delta,
                "cer_base": b.get("cer"),
                "cer_lora": t.get("cer"),
                "improved": improved,
            }
        )

    ndizi_base = _weighted_wer(base_sets, _is_ndizi)
    ndizi_tuned = _weighted_wer(tuned_sets, _is_ndizi)
    ndizi_delta = None
    if ndizi_base and ndizi_tuned:
        ndizi_delta = float(ndizi_tuned["wer"]) - float(ndizi_base["wer"])

    pooled_base = (base_metrics or {}).get("pooled") or {}
    pooled_tuned = (tuned_metrics or {}).get("pooled") or {}
    pooled_delta = None
    if pooled_base.get("wer") is not None and pooled_tuned.get("wer") is not None:
        pooled_delta = float(pooled_tuned["wer"]) - float(pooled_base["wer"])

    comparison = {
        "per_set": rows,
        "ndizi": {
            "wer_base": None if not ndizi_base else ndizi_base["wer"],
            "wer_lora": None if not ndizi_tuned else ndizi_tuned["wer"],
            "delta_wer": ndizi_delta,
            "n": None if not ndizi_tuned else ndizi_tuned["n"],
            "improved": ndizi_delta is not None and ndizi_delta < -1e-6,
        },
        "pooled": {
            "wer_base": pooled_base.get("wer"),
            "wer_lora": pooled_tuned.get("wer"),
            "delta_wer": pooled_delta,
        },
        "note": "delta_wer = LoRA − base; negative means the adapter improved (lower WER).",
    }

    print(f"\n{'═'*72}")
    print("ASR COMPARISON  (WER; lower is better.  Δ = LoRA − base; negative = improved)")
    print(f"{'═'*72}")
    print(f"{'set':<42} {'n':>5} {'base':>7} {'LoRA':>7} {'ΔWER':>8}  {'Ndizi?'}")
    print("-" * 72)
    for row in rows:
        tag = "yes" if row["ndizi"] else ""
        d = "   n/a" if row["delta_wer"] is None else f"{row['delta_wer']:+7.3f}"
        mark = "  ← improved" if row["improved"] else ("  ← worse" if row["delta_wer"] and row["delta_wer"] > 1e-6 else "")
        print(
            f"{row['set']:<42} {str(row['n'] or ''):>5} "
            f"{_fmt(row['wer_base'])} {_fmt(row['wer_lora'])} {d}  {tag}{mark}"
        )
    print("-" * 72)
    nd = comparison["ndizi"]
    d = "   n/a" if nd["delta_wer"] is None else f"{nd['delta_wer']:+7.3f}"
    verdict = "IMPROVED on Ndizi" if nd["improved"] else (
        "NO Ndizi gain" if nd["delta_wer"] is not None else "n/a"
    )
    print(
        f"{'Ndizi pooled (in-domain)':<42} {str(nd['n'] or ''):>5} "
        f"{_fmt(nd['wer_base'])} {_fmt(nd['wer_lora'])} {d}  {verdict}"
    )
    print(f"{'All sets pooled':<42} {'':>5} {_fmt(comparison['pooled']['wer_base'])} "
          f"{_fmt(comparison['pooled']['wer_lora'])} "
          f"{'   n/a' if pooled_delta is None else f'{pooled_delta:+7.3f}'}")
    print()
    return comparison


def _run_variant(args, *, no_adapter: bool, out_dir: Path, run_evaluate, label: str) -> tuple[dict | None, dict | None]:
    print(f"\n{'█'*72}\n  {label}\n{'█'*72}")
    variant = SimpleNamespace(**vars(args))
    variant.no_adapter = no_adapter
    variant.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asr_metrics = None
    if not args.no_asr:
        run_evaluate(variant)
        metrics_file = out_dir / "metrics.json"
        if metrics_file.exists():
            asr_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        _free_cuda()

    chat_summary = None
    if not args.no_chat:
        model, processor = load_eval_model(variant)
        chat_summary = run_chat_eval(variant, model, processor, out_dir)
        del model, processor
        _free_cuda()
    return asr_metrics, chat_summary


def main() -> int:
    load_env_file(ROOT / ".env")
    os.chdir(ROOT)

    from src.eval.finetuned import run_evaluate
    from src.eval.normalize import TEXT_NORMALIZE_EVAL_DEFAULT, add_normalize_arg
    from src.utils.constants import (
        ASR_PROMPT_MAP,
        SUNFLOWER_EVAL_ASR,
        SUNFLOWER_MODEL_ID,
        SUNFLOWER_SYSTEM_PROMPT,
        resolve_asr_prompt,
    )
    from src.utils.paths import SUNFLOWER_CHECKPOINT_DIR
    from src.utils.runtime import apply_model_choice

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=SUNFLOWER_MODEL_ID, help="Sunflower (or other) multimodal base")
    p.add_argument("--checkpoint", default=str(SUNFLOWER_CHECKPOINT_DIR / "best"))
    p.add_argument("--no-adapter", action="store_true", help="Evaluate the base Sunflower weights only")
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip the base-model pass (checkpoint only). Default is base + checkpoint.",
    )
    p.add_argument(
        "--test-datasets",
        nargs="+",
        default=list(SUNFLOWER_EVAL_ASR),
        help="Hub specs: repo:split or repo:config:split. Default Ndizi test + FLEURS sw_ke test.",
    )
    p.add_argument("--no-asr", action="store_true", help="Skip ASR WER (chat only)")
    p.add_argument("--no-chat", action="store_true", help="Skip chat prompts (ASR only). Use if VRAM is tight.")
    p.add_argument("--audio-column", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--chat-prompts", default=str(ROOT / "data" / "sunflower_chat_eval.jsonl"))
    p.add_argument("--chat-prompt-list", nargs="+", default=None)
    p.add_argument("--system-prompt", default=SUNFLOWER_SYSTEM_PROMPT)
    p.add_argument("--output-dir", default="eval/sunflower-ndizi")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--chunk_length_s", type=float, default=30.0)
    p.add_argument(
        "--asr-prompt",
        choices=tuple(ASR_PROMPT_MAP),
        default="ondevice",
        help="Must match training and LiteRT: ondevice = 'Andika maneno unayosikia katika sauti hii.'",
    )
    add_normalize_arg(p, default=TEXT_NORMALIZE_EVAL_DEFAULT)
    args = p.parse_args()
    args.asr_instruction = resolve_asr_prompt(args.asr_prompt)
    print(f"[eval] ASR prompt: {args.asr_prompt} ({args.asr_instruction})")
    args.no_auto_chunk = False
    args.stride_length_s = None
    args.max_audio_seconds = None
    args.retention_eval = False
    args.baseline = False

    apply_model_choice(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_asr and args.no_chat:
        raise SystemExit("Nothing to run: both --no-asr and --no-chat")
    compare = not args.no_adapter and not args.skip_base
    if not args.no_adapter and not Path(args.checkpoint).is_dir():
        raise SystemExit(
            f"LoRA checkpoint not found: {args.checkpoint}\n"
            "Train first, or pass --no-adapter to score the base Sunflower weights."
        )

    base_asr = base_chat = tuned_asr = tuned_chat = None
    if compare:
        base_asr, base_chat = _run_variant(
            args, no_adapter=True, out_dir=out_dir / "base", run_evaluate=run_evaluate,
            label=f"BASE  {args.model}",
        )
        tuned_asr, tuned_chat = _run_variant(
            args, no_adapter=False, out_dir=out_dir / "checkpoint", run_evaluate=run_evaluate,
            label=f"LORA  {args.checkpoint}",
        )
        comparison = compare_asr(base_asr, tuned_asr) if not args.no_asr else None
        if comparison:
            (out_dir / "comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("Wrote", out_dir / "comparison.json")
    elif args.no_adapter:
        base_asr, base_chat = _run_variant(
            args, no_adapter=True, out_dir=out_dir, run_evaluate=run_evaluate,
            label=f"BASE  {args.model}",
        )
        comparison = None
    else:
        tuned_asr, tuned_chat = _run_variant(
            args, no_adapter=False, out_dir=out_dir, run_evaluate=run_evaluate,
            label=f"LORA  {args.checkpoint}",
        )
        comparison = None

    def _chat_brief(s):
        return None if s is None else {k: s[k] for k in ("n", "n_pass", "pass_rate")}

    payload = {
        "model": args.model,
        "checkpoint": None if args.no_adapter else args.checkpoint,
        "test_datasets": args.test_datasets,
        "max_samples": args.max_samples,
        "compared": compare,
        "base": {"asr": base_asr, "chat": _chat_brief(base_chat)},
        "lora": {"asr": tuned_asr, "chat": _chat_brief(tuned_chat)},
        "comparison": comparison,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", summary_path)

    chat_for_gate = tuned_chat or base_chat
    if chat_for_gate is not None and chat_for_gate["pass_rate"] < 0.5:
        print("[warn] chat pass_rate < 0.5 — consider asr_moderate, lower merge scale, or more --chat-ratio")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
