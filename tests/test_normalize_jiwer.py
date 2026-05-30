"""jiwer_default transform must work with jiwer 3.x (SubstituteRegexes dict API)."""
from __future__ import annotations

from src.eval.normalize import pooled_wer_cer, try_build_jiwer_transforms


def test_jiwer_default_does_not_glue_punctuation():
    tr_w, tr_c = try_build_jiwer_transforms()
    assert tr_w is not None and tr_c is not None
    wer, cer = pooled_wer_cer(
        ["hii hapana"],
        ["hii.Hapana"],
        "jiwer_default",
        jiwer_tr_w=tr_w,
        jiwer_tr_c=tr_c,
    )
    assert wer is not None and wer < 0.5


def test_substitute_regexes_builds():
    try_build_jiwer_transforms()
