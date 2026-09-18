#!/usr/bin/env python3
"""Linear soup of two African ASR LoRA+projector checkpoints.

  θ = α · A + (1 − α) · B

Same PEFT layout required (v1 and v2 both asr_moderate on Sunflower).

  python scripts/merge_african_asr_adapters.py \\
    --adapter-a artifacts/checkpoints_african_asr/best \\
    --adapter-b artifacts/checkpoints_african_asr_v2/best \\
    --alpha 0.7 \\
    --output-dir artifacts/checkpoints_african_asr_soup_a07

Sweep several alphas, then gate each with scripts/gate_african_asr_checkpoint.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_tensors(path: Path) -> dict:
    from safetensors.torch import load_file

    if not path.is_file():
        raise FileNotFoundError(path)
    return load_file(str(path))


def _merge_state(a: dict, b: dict, alpha: float) -> dict:
    keys = sorted(set(a) | set(b))
    missing_a = [k for k in keys if k not in a]
    missing_b = [k for k in keys if k not in b]
    if missing_a or missing_b:
        raise SystemExit(
            f"Key mismatch.\n  only in A ({len(missing_a)}): {missing_a[:8]}\n"
            f"  only in B ({len(missing_b)}): {missing_b[:8]}"
        )
    out = {}
    for k in keys:
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            raise SystemExit(f"Shape mismatch on {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        out[k] = (alpha * ta.float() + (1.0 - alpha) * tb.float()).to(ta.dtype)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter-a", required=True, help="First adapter dir (e.g. v1 best)")
    p.add_argument("--adapter-b", required=True, help="Second adapter dir (e.g. v2 best)")
    p.add_argument("--alpha", type=float, default=0.7, help="Weight on A (default 0.7 → favor v1)")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    a_dir = Path(args.adapter_a)
    b_dir = Path(args.adapter_b)
    out = Path(args.output_dir)
    alpha = float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    for d in (a_dir, b_dir):
        if not (d / "adapter_config.json").is_file():
            raise SystemExit(f"Missing adapter_config.json in {d}")

    out.mkdir(parents=True, exist_ok=True)
    # Prefer A's config/tokenizer files; weights are merged.
    for name in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "preprocessor_config.json", "processor_config.json", "chat_template.jinja", "training_mode.json"):
        src = a_dir / name
        if src.is_file():
            shutil.copy2(src, out / name)

    from safetensors.torch import save_file

    # LoRA weights (PEFT may use adapter_model.safetensors or .bin)
    a_lora = a_dir / "adapter_model.safetensors"
    b_lora = b_dir / "adapter_model.safetensors"
    if a_lora.is_file() and b_lora.is_file():
        merged = _merge_state(_load_tensors(a_lora), _load_tensors(b_lora), alpha)
        save_file(merged, out / "adapter_model.safetensors")
        print(f"[soup] merged LoRA tensors: {len(merged)}")
    else:
        raise SystemExit(
            f"Need adapter_model.safetensors in both dirs.\n  A={a_lora.is_file()} B={b_lora.is_file()}"
        )

    a_proj = a_dir / "projector_weights.safetensors"
    b_proj = b_dir / "projector_weights.safetensors"
    if a_proj.is_file() and b_proj.is_file():
        merged_p = _merge_state(_load_tensors(a_proj), _load_tensors(b_proj), alpha)
        save_file(merged_p, out / "projector_weights.safetensors")
        print(f"[soup] merged projector tensors: {len(merged_p)}")
    elif a_proj.is_file() or b_proj.is_file():
        print("[warn] projector_weights only on one side — copying from the side that has it")
        shutil.copy2(a_proj if a_proj.is_file() else b_proj, out / "projector_weights.safetensors")

    meta = {
        "adapter_a": str(a_dir.resolve()),
        "adapter_b": str(b_dir.resolve()),
        "alpha": alpha,
        "formula": "alpha * A + (1 - alpha) * B",
    }
    (out / "soup_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[soup] wrote {out}  (α={alpha:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
