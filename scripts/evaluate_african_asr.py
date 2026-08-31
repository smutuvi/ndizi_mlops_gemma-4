#!/usr/bin/env python3
"""Score Google Gemma 4 E2B on every African ASR test set in one run.

Same eval as:

  python scripts/evaluate_sunflower_swahili.py \\
    --model google/gemma-4-e2b-it \\
    --no-adapter \\
    --test-datasets \\
      smutuvi/ndizi-1:test \\
      smutuvi/ndizi-1-2025:test \\
      google/fleurs:sw_ke:test \\
      google/WaxalNLP:amh_asr:test \\
      google/WaxalNLP:orm_asr:test \\
      turiabu/Sagalee:test \\
    --output-dir eval/google-gemma4-e2b-base \\
    --no-chat

Default prompt is ``auto`` (Swahili / Amharic / Oromo by dataset). Pass
``--asr-prompt ondevice`` to force the Swahili LiteRT line on every set.

  python scripts/evaluate_african_asr.py
  python scripts/evaluate_african_asr.py --max-samples 32
  python scripts/evaluate_african_asr.py --asr-prompt ondevice
  python scripts/evaluate_african_asr.py --compare --checkpoint artifacts/checkpoints_african_asr/best
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

DEFAULT_TEST_DATASETS = [
    "smutuvi/ndizi-1:test",
    "smutuvi/ndizi-1-2025:test",
    "google/fleurs:sw_ke:test",
    "google/WaxalNLP:amh_asr:test",
    "google/WaxalNLP:orm_asr:test",
    "turiabu/Sagalee:test",
]
SAGALEE_TEST = "turiabu/Sagalee:test"


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


def _fmt(x) -> str:
    if x is None:
        return "   n/a"
    return f"{float(x):6.3f}"


def _per_set(metrics: dict | None) -> dict:
    return (metrics or {}).get("per_set") or {}


def _print_table(google: dict | None, sunflower: dict | None, lora: dict | None) -> None:
    g, s, t = _per_set(google), _per_set(sunflower), _per_set(lora)
    names = list(dict.fromkeys([*g, *s, *t]))
    if not names:
        return
    cols = [("Google", g), ("Sunfl.", s), ("LoRA", t)]
    active = [(label, rows) for label, rows in cols if rows]
    print(f"\n{'═' * 88}")
    print("ASR  (WER; lower is better)")
    print(f"{'═' * 88}")
    header = f"{'set':<42} {'n':>5}" + "".join(f" {label:>7}" for label, _ in active)
    if sunflower and lora:
        header += f" {'ΔL−S':>8}"
    print(header)
    print("-" * 88)
    for name in names:
        gg, ss, tt = g.get(name) or {}, s.get(name) or {}, t.get(name) or {}
        n = tt.get("n") or ss.get("n") or gg.get("n") or ""
        line = f"{name:<42} {str(n):>5}"
        for label, rows in active:
            line += f" {_fmt((rows.get(name) or {}).get('wer'))}"
        if sunflower and lora:
            sw, tw = ss.get("wer"), tt.get("wer")
            delta = None if sw is None or tw is None else float(tw) - float(sw)
            line += "    n/a" if delta is None else f" {delta:+7.3f}"
        print(line)
    print("-" * 88)
    print()


def _run_system(*, label: str, model: str, no_adapter: bool, checkpoint: str | None, out_dir: Path, base_args) -> dict | None:
    from src.eval.finetuned import run_evaluate
    from src.utils.runtime import apply_model_choice

    print(f"\n{'█' * 72}\n  {label}\n{'█' * 72}")
    apply_model_choice(model)
    ns = SimpleNamespace(**vars(base_args))
    ns.no_adapter = no_adapter
    ns.checkpoint = checkpoint
    ns.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_evaluate(ns)
    _free_cuda()
    metrics_file = out_dir / "metrics.json"
    if not metrics_file.is_file():
        return None
    return json.loads(metrics_file.read_text(encoding="utf-8"))


def main() -> int:
    load_env_file(ROOT / ".env")
    os.chdir(ROOT)

    from src.eval.normalize import TEXT_NORMALIZE_EVAL_DEFAULT, add_normalize_arg
    from src.utils.constants import ASR_PROMPT_MAP, LANG_ASR_PROMPTS, SUNFLOWER_MODEL_ID, resolve_asr_prompt
    from src.utils.paths import AFRICAN_ASR_CHECKPOINT_DIR

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/gemma-4-e2b-it", help="Google (or other) base for the first pass")
    p.add_argument("--sunflower-model", default=SUNFLOWER_MODEL_ID)
    p.add_argument("--checkpoint", default=str(AFRICAN_ASR_CHECKPOINT_DIR / "best"))
    p.add_argument("--test-datasets", nargs="+", default=list(DEFAULT_TEST_DATASETS))
    p.add_argument("--include-sagalee", action="store_true", help=f"Append {SAGALEE_TEST}")
    p.add_argument("--compare", action="store_true", help="Also score Sunflower base and the Ndizi LoRA")
    p.add_argument("--with-sunflower", action="store_true")
    p.add_argument("--with-lora", action="store_true")
    p.add_argument("--skip-google", action="store_true")
    p.add_argument("--output-dir", default="eval/google-gemma4-e2b-base")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--chunk_length_s", type=float, default=30.0)
    p.add_argument("--audio-column", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--dataset-revision", default=None)
    p.add_argument(
        "--asr-prompt",
        choices=(*ASR_PROMPT_MAP, "auto"),
        default="auto",
        help="auto = Swahili/Amharic/Oromo by dataset. ondevice = Swahili LiteRT line on every set.",
    )
    add_normalize_arg(p, default=TEXT_NORMALIZE_EVAL_DEFAULT)
    args = p.parse_args()

    if args.include_sagalee and SAGALEE_TEST not in args.test_datasets:
        args.test_datasets = [*args.test_datasets, SAGALEE_TEST]

    run_google = not args.skip_google
    run_sunflower = args.compare or args.with_sunflower
    run_lora = args.compare or args.with_lora
    n_systems = int(run_google) + int(run_sunflower) + int(run_lora)
    if n_systems == 0:
        raise SystemExit("Nothing to run: skipped google and did not request sunflower/lora.")
    if run_lora and not Path(args.checkpoint).is_dir():
        raise SystemExit(f"LoRA checkpoint not found: {args.checkpoint}")

    if args.asr_prompt == "auto":
        args.asr_instruction = LANG_ASR_PROMPTS["sw"]
        print("[eval] ASR prompt: auto (Swahili / Amharic / Oromo by dataset)")
    else:
        args.asr_instruction = resolve_asr_prompt(args.asr_prompt)
        print(f"[eval] ASR prompt: {args.asr_prompt} ({args.asr_instruction})")
    args.no_asr = False
    args.no_chat = True
    args.no_auto_chunk = False
    args.stride_length_s = None
    args.max_audio_seconds = None
    args.retention_eval = False
    args.baseline = False
    print("[eval] sets:", " ".join(args.test_datasets))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nest = n_systems > 1

    google_m = sunflower_m = lora_m = None
    if run_google:
        google_m = _run_system(
            label=f"GOOGLE  {args.model}",
            model=args.model,
            no_adapter=True,
            checkpoint=None,
            out_dir=out / "google" if nest else out,
            base_args=args,
        )
    if run_sunflower:
        sunflower_m = _run_system(
            label=f"SUNFLOWER  {args.sunflower_model}",
            model=args.sunflower_model,
            no_adapter=True,
            checkpoint=None,
            out_dir=out / "sunflower" if nest else out,
            base_args=args,
        )
    if run_lora:
        lora_m = _run_system(
            label=f"LORA  {args.checkpoint}",
            model=args.sunflower_model,
            no_adapter=False,
            checkpoint=args.checkpoint,
            out_dir=out / "lora" if nest else out,
            base_args=args,
        )

    _print_table(google_m, sunflower_m, lora_m)
    payload = {
        "test_datasets": args.test_datasets,
        "asr_prompt": args.asr_prompt,
        "max_samples": args.max_samples,
        "google_model": None if not run_google else args.model,
        "sunflower_model": None if not run_sunflower else args.sunflower_model,
        "checkpoint": None if not run_lora else args.checkpoint,
        "google": google_m,
        "sunflower": sunflower_m,
        "lora": lora_m,
    }
    summary = out / ("comparison.json" if nest else "summary.json")
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
