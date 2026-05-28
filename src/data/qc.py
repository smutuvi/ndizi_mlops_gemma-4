# src/data/qc.py — optional multi-gate audio/text QC (ported from ndizi_mlops).
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

_SWAHILI_FUNCTION_WORDS = {
    "na",
    "ya",
    "kwa",
    "katika",
    "kama",
    "hii",
    "hiyo",
    "hizo",
    "hapa",
    "pale",
    "ni",
    "sio",
    "ndio",
    "kuwa",
    "yenye",
    "bila",
    "sana",
    "tu",
    "pia",
    "lakini",
    "hivyo",
    "ambayo",
    "hapo",
    "huku",
    "kile",
    "mimi",
    "sisi",
    "yeye",
    "wao",
    "wangu",
    "wetu",
    "wake",
    "zao",
    "kwangu",
    "kwetu",
    "kwake",
    "kwenu",
}
_SW_PREFIX_RE = re.compile(r"^(ku|wa|ni|ya|ki|vi|li|si|ha|na)\w+$")


@dataclass
class QCConfig:
    # audio
    min_dur: float = 1.0
    max_dur: float = 30.0
    min_rms_dbfs: float = -45.0
    clip_thresh: float = 0.999
    max_clipping_rate: float = 0.002
    vad_frame_ms: float = 25.0
    vad_hop_ms: float = 10.0
    vad_db_thresh: float = -35.0
    min_speech_ratio: float = 0.25
    # text
    min_text_chars: int = 3
    max_text_chars: int = 400
    max_weird_char_ratio: float = 0.02
    max_repetition_token_prop: float = 0.35
    max_digit_ratio: float = 0.20
    min_swahili_likeness: float = 0.05
    min_tokens_for_langcheck: int = 6
    min_words_per_sec: float = 0.5
    max_words_per_sec: float = 4.5
    min_chars_per_sec: float = 4.0
    max_chars_per_sec: float = 25.0


def _rms_dbfs(audio: np.ndarray) -> float:
    eps = 1e-12
    rms = float(np.sqrt(np.mean(audio * audio) + eps))
    return 20.0 * math.log10(max(rms, eps))


def _clipping_rate(audio: np.ndarray, clip_thresh: float) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= clip_thresh))


def _energy_vad_speech_ratio(
    audio: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    db_thresh: float,
) -> float:
    if len(audio) == 0 or sr <= 0:
        return 0.0
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if frame <= 0 or hop <= 0 or len(audio) < frame:
        return 1.0 if _rms_dbfs(audio) > db_thresh else 0.0
    eps = 1e-12
    n_frames = 1 + (len(audio) - frame) // hop
    speech_frames = 0
    for i in range(n_frames):
        seg = audio[i * hop : i * hop + frame]
        seg_rms = float(np.sqrt(np.mean(seg * seg) + eps))
        if 20.0 * math.log10(max(seg_rms, eps)) > db_thresh:
            speech_frames += 1
    return float(speech_frames) / float(n_frames)


def _char_stats(text: str) -> Dict[str, float]:
    if not text:
        return {"digit_ratio": 0.0, "weird_ratio": 0.0}
    n = len(text)
    digits = sum(ch.isdigit() for ch in text)
    weird = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ("'", "-", "_")))
    return {"digit_ratio": digits / n, "weird_ratio": weird / n}


def _repetition_score(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) / len(toks)


def _swahili_likeness(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _SWAHILI_FUNCTION_WORDS)
    prefix_hits = sum(1 for t in toks if _SW_PREFIX_RE.match(t))
    return (hits + 0.25 * prefix_hits) / len(toks)


def evaluate_example(
    audio_array: np.ndarray,
    sr: int,
    text: str,
    cfg: QCConfig,
) -> Dict[str, Any]:
    audio = np.asarray(audio_array, dtype=np.float32)
    dur = len(audio) / sr if sr > 0 else 0.0
    rms = _rms_dbfs(audio)
    clip = _clipping_rate(audio, cfg.clip_thresh)
    vad = _energy_vad_speech_ratio(audio, sr, cfg.vad_frame_ms, cfg.vad_hop_ms, cfg.vad_db_thresh)
    text_s = (text or "").strip()
    stats = _char_stats(text_s)
    toks = text_s.split()
    wps = len(toks) / dur if dur > 0 else 0.0
    cps = len(text_s) / dur if dur > 0 else 0.0
    rep = _repetition_score(text_s)
    sw = _swahili_likeness(text_s)

    def fail(reason: str) -> Dict[str, Any]:
        return {"keep": False, "qc_reason": reason, "duration_sec": dur}

    if dur < cfg.min_dur:
        return fail("dur_low")
    if dur > cfg.max_dur:
        return fail("dur_high")
    if rms < cfg.min_rms_dbfs:
        return fail("rms_low")
    if clip > cfg.max_clipping_rate:
        return fail("clip_high")
    if vad < cfg.min_speech_ratio:
        return fail("vad_low")
    if len(text_s) < cfg.min_text_chars:
        return fail("text_short")
    if len(text_s) > cfg.max_text_chars:
        return fail("text_long")
    if stats["weird_ratio"] > cfg.max_weird_char_ratio:
        return fail("weird_high")
    if rep > cfg.max_repetition_token_prop:
        return fail("repeat_high")
    if stats["digit_ratio"] > cfg.max_digit_ratio:
        return fail("digit_high")
    if len(toks) >= cfg.min_tokens_for_langcheck and sw < cfg.min_swahili_likeness:
        return fail("swahili_low")
    if wps < cfg.min_words_per_sec:
        return fail("wps_low")
    if wps > cfg.max_words_per_sec:
        return fail("wps_high")
    if cps < cfg.min_chars_per_sec:
        return fail("cps_low")
    if cps > cfg.max_chars_per_sec:
        return fail("cps_high")

    return {"keep": True, "qc_reason": "ok", "duration_sec": dur}


def check_example(audio: dict, text: str, cfg: QCConfig) -> Tuple[bool, str]:
    a = audio.get("array")
    sr = int(audio.get("sampling_rate") or 0)
    if a is None or sr <= 0:
        return False, "missing_audio"
    out = evaluate_example(np.asarray(a, dtype=np.float32), sr, text, cfg)
    return bool(out["keep"]), str(out["qc_reason"])

