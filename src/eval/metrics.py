# src/eval/metrics.py — pooled and grouped WER/CER evaluation helpers.
from __future__ import annotations

from tqdm.auto import tqdm

from src.eval.normalize import TEXT_NORMALIZE_DEFAULT, pooled_wer_cer, try_build_jiwer_transforms
from src.utils.constants import AUDIO_COLUMN, TEXT_COLUMN


def eval_with(predict_fn, ds, batch_size=4, group_col="source_dataset", desc="evaluating"):
    refs, hyps, groups, buf = [], [], [], []
    has_group = group_col in ds.column_names
    pbar = tqdm(total=len(ds), desc=desc, unit="clip", dynamic_ncols=True)
    for row in ds:
        buf.append(row)
        if len(buf) == batch_size:
            refs += [r[TEXT_COLUMN] for r in buf]
            hyps += predict_fn([r[AUDIO_COLUMN] for r in buf])
            if has_group:
                groups += [r[group_col] for r in buf]
            pbar.update(len(buf))
            buf = []
    if buf:
        refs += [r[TEXT_COLUMN] for r in buf]
        hyps += predict_fn([r[AUDIO_COLUMN] for r in buf])
        if has_group:
            groups += [r[group_col] for r in buf]
        pbar.update(len(buf))
    pbar.close()
    return refs, hyps, groups


def score_grouped(refs, hyps, groups, normalize: str = TEXT_NORMALIZE_DEFAULT):
    """Pooled WER/CER plus per-``source_dataset`` breakdown (e.g. smutuvi/ndizi-1, smutuvi/ndizi-1-2025)."""
    jiwer_tr_w, jiwer_tr_c = None, None
    if normalize == "jiwer_default":
        jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()

    wer, cer = pooled_wer_cer(
        hyps, refs, normalize, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c
    )
    out = {
        "pooled": {"wer": wer, "cer": cer, "n": len(refs), "normalize": normalize},
    }
    if groups:
        for g in sorted(set(groups)):
            gr = [r for r, gv in zip(refs, groups) if gv == g]
            gh = [h for h, gv in zip(hyps, groups) if gv == g]
            if gr:
                gw, gc = pooled_wer_cer(
                    gh, gr, normalize, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c
                )
                out[g] = {"wer": gw, "cer": gc, "n": len(gr), "normalize": normalize}
    return out


def print_grouped(label, scores, normalize: str | None = None):
    norm = normalize or scores.get("pooled", {}).get("normalize", TEXT_NORMALIZE_DEFAULT)
    print(f"\n[{label}]  normalize={norm}")
    for g, m in scores.items():
        if g == "normalize":
            continue
        wer = m.get("wer")
        cer = m.get("cer")
        wer_s = f"{wer:.3f}" if wer is not None else "n/a"
        cer_s = f"{cer:.3f}" if cer is not None else "n/a"
        print(f"  {g:>30}  WER={wer_s}  CER={cer_s}  n={m['n']}")
