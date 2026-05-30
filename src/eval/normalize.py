# src/eval/normalize.py — WER/CER text normalization (aligned with ndizi_mlops evaluate_asr_batch).
from __future__ import annotations

from typing import Any, Dict

TEXT_NORMALIZE_CHOICES = ("none", "simple", "jiwer_default")
TEXT_NORMALIZE_DEFAULT = "jiwer_default"  # baseline / programmatic fallback
TEXT_NORMALIZE_EVAL_DEFAULT = "none"  # evaluate_gemma4.py and run_pipeline.py evaluate


def add_normalize_arg(parser, *, default: str = TEXT_NORMALIZE_DEFAULT) -> None:
    parser.add_argument(
        "--normalize",
        choices=TEXT_NORMALIZE_CHOICES,
        default=default,
        help="WER/CER normalization: none (raw), simple (lower+whitespace), "
        "jiwer_default (lower, strip, remove punctuation; matches ndizi_mlops batch eval)",
    )


def simple_normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def try_build_jiwer_transforms() -> tuple[Any, Any]:
    import jiwer

    # jiwer >=3.3 expects a dict for SubstituteRegexes (not a list of pairs).
    # Substitute punctuation spans with space so "hii.Hapana" -> "hii hapana" (not "hiihapana").
    punct_to_space = jiwer.SubstituteRegexes({r"[^\w\s']+": " "})
    tr_w = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            punct_to_space,
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    tr_c = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            punct_to_space,
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemoveWhiteSpace(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )
    return tr_w, tr_c


def pooled_wer_cer(
    preds: list[str],
    refs: list[str],
    mode: str = TEXT_NORMALIZE_DEFAULT,
    *,
    jiwer_tr_w: Any = None,
    jiwer_tr_c: Any = None,
) -> tuple[float | None, float | None]:
    pairs = [(p, r) for p, r in zip(preds, refs) if str(r).strip()]
    if not pairs:
        return None, None
    pl, rl = [p for p, _ in pairs], [r for _, r in pairs]

    import jiwer

    if mode == "simple":
        pl = [simple_normalize(p) for p in pl]
        rl = [simple_normalize(r) for r in rl]
        return float(jiwer.wer(rl, pl)), float(jiwer.cer(rl, pl))

    if mode == "jiwer_default":
        if jiwer_tr_w is None or jiwer_tr_c is None:
            jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
        return (
            float(jiwer.wer(rl, pl, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w)),
            float(jiwer.cer(rl, pl, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c)),
        )

    return float(jiwer.wer(rl, pl)), float(jiwer.cer(rl, pl))


def utterance_wer_cer(
    ref: str,
    hyp: str,
    mode: str = TEXT_NORMALIZE_DEFAULT,
    *,
    jiwer_tr_w: Any = None,
    jiwer_tr_c: Any = None,
) -> tuple[float, float]:
    w, c = pooled_wer_cer([hyp], [ref], mode, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c)
    return float(w or 0.0), float(c or 0.0)


def extra_normalized_fields_for_row(
    ref_raw: str,
    pred_raw: str,
    mode: str,
    *,
    jiwer_tr_w: Any = None,
    jiwer_tr_c: Any = None,
) -> Dict[str, Any]:
    """ndizi_mlops-compatible per-row normalized fields (reference/prediction stay raw)."""
    if mode == "none":
        return {}
    if mode == "simple":
        wn, cn = utterance_wer_cer(ref_raw, pred_raw, mode, jiwer_tr_w=jiwer_tr_w, jiwer_tr_c=jiwer_tr_c)
        return {
            "text_normalized": simple_normalize(ref_raw),
            "prediction_normalized": simple_normalize(pred_raw),
            "wer_normalized": wn,
            "cer_normalized": cn,
        }

    import jiwer

    if jiwer_tr_w is None or jiwer_tr_c is None:
        jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()
    wo = jiwer.process_words(
        ref_raw, pred_raw, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w
    )
    co = jiwer.process_characters(
        ref_raw, pred_raw, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c
    )
    rn = " ".join(wo.references[0]) if wo.references and wo.references[0] else ""
    pn = " ".join(wo.hypotheses[0]) if wo.hypotheses and wo.hypotheses[0] else ""
    return {
        "text_normalized": rn,
        "prediction_normalized": pn,
        "wer_normalized": float(wo.wer),
        "cer_normalized": float(co.cer),
    }
