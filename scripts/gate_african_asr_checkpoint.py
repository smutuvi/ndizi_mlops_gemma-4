#!/usr/bin/env python3
"""Hard gate: LoRA must not regress vs Sunflower on Must sets.

Works offline on a comparison.json from evaluate_african_asr.py, or runs eval
then gates.

  # Offline (you already have comparison.json from v3):
  python scripts/gate_african_asr_checkpoint.py \\
    --comparison /Users/mutuvi/Downloads/comparison.json

  # Run eval then gate:
  python scripts/gate_african_asr_checkpoint.py \\
    --checkpoint artifacts/checkpoints_african_asr_v4/best \\
    --output-dir eval/african_asr_v4_gate

Must (default): Ndizi-1, Ndizi-1-2025, FLEURS sw_ke  →  Δ = LoRA − Sunflower ≤ tol
Should (default): Waxal orm, Sagalee                 →  reported, not required
Ndizi pooled must also improve by at least --min-ndizi-delta (default −0.10).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _wer(block: dict | None, name: str) -> float | None:
    if not block:
        return None
    row = (block.get("per_set") or {}).get(name) or {}
    w = row.get("wer")
    return None if w is None else float(w)


def _ndizi_pooled(block: dict | None) -> tuple[float | None, int]:
    if not block:
        return None, 0
    per = block.get("per_set") or {}
    rows = []
    for key in ("smutuvi/ndizi-1:test", "smutuvi/ndizi-1-2025:test"):
        r = per.get(key) or {}
        if r.get("wer") is not None and r.get("n"):
            rows.append((float(r["wer"]), int(r["n"])))
    if not rows:
        return None, 0
    n = sum(n for _, n in rows)
    return sum(w * n for w, n in rows) / n, n


def gate(
    sunflower: dict,
    lora: dict,
    *,
    must: list[str],
    should: list[str],
    tol: float,
    min_ndizi_delta: float,
) -> dict:
    rows = []
    must_fail = []
    for name in must:
        sw, lw = _wer(sunflower, name), _wer(lora, name)
        if sw is None or lw is None:
            must_fail.append(name)
            rows.append({"set": name, "tier": "must", "sunflower": sw, "lora": lw, "delta": None, "ok": False, "note": "missing"})
            continue
        delta = lw - sw
        ok = delta <= tol
        if not ok:
            must_fail.append(name)
        rows.append({"set": name, "tier": "must", "sunflower": sw, "lora": lw, "delta": delta, "ok": ok, "note": ""})

    should_fail = []
    for name in should:
        sw, lw = _wer(sunflower, name), _wer(lora, name)
        if sw is None or lw is None:
            rows.append({"set": name, "tier": "should", "sunflower": sw, "lora": lw, "delta": None, "ok": False, "note": "missing"})
            should_fail.append(name)
            continue
        delta = lw - sw
        ok = delta <= tol
        if not ok:
            should_fail.append(name)
        rows.append({"set": name, "tier": "should", "sunflower": sw, "lora": lw, "delta": delta, "ok": ok, "note": ""})

    sw_p, n = _ndizi_pooled(sunflower)
    lw_p, _ = _ndizi_pooled(lora)
    ndizi_delta = None if sw_p is None or lw_p is None else lw_p - sw_p
    ndizi_ok = ndizi_delta is not None and ndizi_delta <= min_ndizi_delta

    passed = not must_fail and ndizi_ok
    return {
        "passed": passed,
        "must_fail": must_fail,
        "should_fail": should_fail,
        "ndizi_pooled": {"sunflower": sw_p, "lora": lw_p, "delta": ndizi_delta, "n": n, "ok": ndizi_ok},
        "tol": tol,
        "min_ndizi_delta": min_ndizi_delta,
        "rows": rows,
    }


def _print(report: dict) -> None:
    print(f"\n{'═' * 88}")
    print("GATE  (Δ = LoRA − Sunflower; pass if Δ ≤ tol on Must + Ndizi pooled ≤ min_ndizi_delta)")
    print(f"{'═' * 88}")
    print(f"{'set':<42} {'tier':<7} {'Sunfl.':>7} {'LoRA':>7} {'Δ':>8}  ok?")
    print("-" * 88)
    for r in report["rows"]:
        sw = "   n/a" if r["sunflower"] is None else f"{r['sunflower']:6.3f}"
        lw = "   n/a" if r["lora"] is None else f"{r['lora']:6.3f}"
        d = "   n/a" if r["delta"] is None else f"{r['delta']:+7.3f}"
        mark = "✓" if r["ok"] else "✗"
        note = f"  {r['note']}" if r.get("note") else ""
        print(f"{r['set']:<42} {r['tier']:<7} {sw} {lw} {d}  {mark}{note}")
    nd = report["ndizi_pooled"]
    d = "   n/a" if nd["delta"] is None else f"{nd['delta']:+7.3f}"
    print("-" * 88)
    print(
        f"{'Ndizi pooled':<42} {'must':<7} "
        f"{nd['sunflower'] if nd['sunflower'] is not None else float('nan'):6.3f} "
        f"{nd['lora'] if nd['lora'] is not None else float('nan'):6.3f} "
        f"{d}  {'✓' if nd['ok'] else '✗'}  "
        f"(need Δ ≤ {report['min_ndizi_delta']:+.3f})"
    )
    print("-" * 88)
    print("PASS" if report["passed"] else "FAIL", end="")
    if report["must_fail"]:
        print(f"  must_fail={report['must_fail']}", end="")
    if report["should_fail"]:
        print(f"  should_warn={report['should_fail']}", end="")
    print()


def main() -> int:
    from src.utils.constants import AFRICAN_ASR_GATE_MUST, AFRICAN_ASR_GATE_SHOULD

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison", default=None, help="Path to evaluate_african_asr comparison.json")
    p.add_argument("--checkpoint", default=None, help="If set (and no --comparison), run eval first")
    p.add_argument("--output-dir", default="eval/african_asr_gate")
    p.add_argument("--tol", type=float, default=0.0, help="Max allowed Δ on Must/Should (default 0 = no worse)")
    p.add_argument(
        "--min-ndizi-delta",
        type=float,
        default=-0.10,
        help="Ndizi pooled Δ must be ≤ this (default -0.10 = at least 10pt gain)",
    )
    p.add_argument("--must", nargs="*", default=list(AFRICAN_ASR_GATE_MUST))
    p.add_argument("--should", nargs="*", default=list(AFRICAN_ASR_GATE_SHOULD))
    args = p.parse_args()

    if args.comparison:
        data = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    elif args.checkpoint:
        out = Path(args.output_dir)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_african_asr.py"),
            "--skip-google",
            "--with-sunflower",
            "--with-lora",
            "--asr-prompt",
            "auto",
            "--checkpoint",
            args.checkpoint,
            "--output-dir",
            str(out),
            "--test-datasets",
            *dict.fromkeys([*args.must, *args.should]),  # unique, preserve order
        ]
        print("[gate] running:", " ".join(cmd))
        env = os.environ.copy()
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)
        comp = out / "comparison.json"
        if not comp.is_file():
            # evaluate_african_asr may nest metrics; build from sunflower/lora metrics.json
            sun = json.loads((out / "sunflower" / "metrics.json").read_text(encoding="utf-8"))
            lora = json.loads((out / "lora" / "metrics.json").read_text(encoding="utf-8"))
            data = {"sunflower": sun, "lora": lora, "checkpoint": args.checkpoint}
            comp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            data = json.loads(comp.read_text(encoding="utf-8"))
    else:
        raise SystemExit("Pass --comparison PATH or --checkpoint DIR")

    sunflower = data.get("sunflower") or {}
    lora = data.get("lora") or {}
    report = gate(
        sunflower,
        lora,
        must=list(args.must),
        should=list(args.should),
        tol=float(args.tol),
        min_ndizi_delta=float(args.min_ndizi_delta),
    )
    _print(report)
    out_path = Path(args.output_dir) / "gate_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
