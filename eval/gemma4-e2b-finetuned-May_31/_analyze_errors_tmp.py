#!/usr/bin/env python3
import csv, re, statistics, json
from collections import Counter, defaultdict
from pathlib import Path

path = Path(__file__).parent / "predictions.csv"
rows = list(csv.DictReader(path.open(encoding="utf-8")))
n = len(rows)
wers = [float(r["wer_normalized"]) for r in rows]

def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def loop_score(text):
    words = text.split()
    best = 0
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and words[j] == words[i]:
            j += 1
        best = max(best, j - i)
        i = j
    for ng in (2, 3, 4):
        for i in range(max(0, len(words) - ng * 3)):
            ng_t = tuple(words[i : i + ng])
            reps = 1
            k = i + ng
            while k + ng <= len(words) and tuple(words[k : k + ng]) == ng_t:
                reps += 1
                k += ng
            best = max(best, reps * ng)
    return best

out = []
out.append(f"Total utterances: {n}")
out.append(f"WER normalized: mean={statistics.mean(wers):.3f} median={statistics.median(wers):.3f} p90={sorted(wers)[int(0.9*n)]:.3f}")

for lo, hi, label in [(0, 0.2, "good"), (0.2, 0.4, "moderate"), (0.4, 0.6, "fair"), (0.6, 0.8, "poor"), (0.8, 1.01, "bad"), (1.01, 99, "catastrophic")]:
    c = sum(1 for w in wers if lo <= w < hi)
    out.append(f"  {label}: {c} ({100*c/n:.1f}%)")

loops = [r for r in rows if loop_score(norm(r["prediction"])) >= 6]
out.append(f"Looping preds: {len(loops)} ({100*len(loops)/n:.1f}%)")

sub_like = Counter()
patterns = [
    (r"mauwa", "maua"), (r"marage", "maharage"), (r"kuusu", "kuhusu"),
    (r"alizi", "ardhi"), (r"kunaa", "kuna"), (r"piya", "pia"),
    (r"sanaa", "sana"), (r"ukungu", "upunguu"), (r"stage", "steji"),
]
for r in rows:
    ref_s, hyp_s = norm(r["reference"]), norm(r["prediction"])
    for rx, hy in patterns:
        if re.search(rx, ref_s) and re.search(hy, hyp_s):
            sub_like[f"{rx}->{hy}"] += len(re.findall(rx, ref_s))

out.append("Lexical shifts:")
for k, v in sub_like.most_common(12):
    out.append(f"  {v:4d}  {k}")

try:
    import jiwer
    from jiwer import Compose, ToLowerCase, RemoveMultipleSpaces, SubstituteRegexes

    tr = Compose([ToLowerCase(), SubstituteRegexes({r"[^\w\s]+": " "}), RemoveMultipleSpaces()])
    subs, ins, dels = Counter(), Counter(), Counter()
    for r in rows:
        ref, hyp = tr(r["reference"]), tr(r["prediction"])
        if not ref:
            continue
        align = jiwer.process_words(ref, hyp)
        for ch in align.alignments[0]:
            if ch.type == "substitute":
                rw = " ".join(ref.split()[ch.ref_start_idx : ch.ref_end_idx])
                hw = " ".join(hyp.split()[ch.hyp_start_idx : ch.hyp_end_idx])
                if rw and hw:
                    subs[(rw, hw)] += 1
            elif ch.type == "insert":
                hw = " ".join(hyp.split()[ch.hyp_start_idx : ch.hyp_end_idx])
                if hw:
                    ins[hw] += 1
            elif ch.type == "delete":
                rw = " ".join(ref.split()[ch.ref_start_idx : ch.ref_end_idx])
                if rw:
                    dels[rw] += 1
    out.append("Top substitutions:")
    for (a, b), c in subs.most_common(25):
        if c >= 8:
            out.append(f"  {c:4d}  {a!r} -> {b!r}")
    out.append("Top insertions:")
    for w, c in ins.most_common(20):
        if c >= 12:
            out.append(f"  {c:4d}  +{w!r}")
    out.append("Top deletions:")
    for w, c in dels.most_common(20):
        if c >= 12:
            out.append(f"  {c:4d}  -{w!r}")
except ImportError:
    out.append("jiwer not installed")

dur = [(float(r["audio_duration_s"]), float(r["wer_normalized"])) for r in rows if r.get("audio_duration_s")]
for label, fn in [("<30s", lambda d: d < 30), ("30-60s", lambda d: 30 <= d < 60), (">=60s", lambda d: d >= 60)]:
    ws = [w for d, w in dur if fn(d)]
    if ws:
        out.append(f"WER {label}: n={len(ws)} mean={statistics.mean(ws):.3f}")

(Path(__file__).parent / "_analysis_out.txt").write_text("\n".join(out), encoding="utf-8")
