# src/data/mms_fa_chunk.py — MMS forced-alignment chunking for clips > Gemma 4 audio limit.
from __future__ import annotations

import re

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm

from src.utils.constants import AUDIO_COLUMN, MAX_AUDIO_SEC, TEXT_COLUMN, TARGET_SR


def add_chunk_index_zero(ds):
    """Schema parity: one row per clip, chunk_index always 0."""
    if "chunk_index" in ds.column_names:
        return ds
    return ds.add_column("chunk_index", [0] * len(ds))


def align_and_chunk_long_clips(
    ds,
    max_chunk_sec: float = 28.0,
    model_id: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    *,
    add_reassembly: bool = False,
    desc: str = "MMS-FA chunk",
):
    """
    Split clips longer than MAX_AUDIO_SEC into <=max_chunk_sec segments using MMS forced alignment.
    """
    try:
        from torchaudio.functional import forced_align
    except ImportError as exc:
        raise RuntimeError(
            "MMS-FA chunking needs torchaudio.functional.forced_align (torchaudio>=2.1). "
            "Upgrade torchaudio or skip --chunk-long-audio."
        ) from exc

    from transformers import AutoProcessor, Wav2Vec2ForCTC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MMS-FA model ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()

    tok = processor.tokenizer
    vocab = tok.get_vocab()
    _blank = getattr(model.config, "pad_token_id", None)
    blank_id = (
        _blank
        if _blank is not None
        else (tok.pad_token_id if tok.pad_token_id is not None else 0)
    )
    sep = "|" if "|" in vocab else " "
    sep_id = vocab.get(sep)

    punct_re = re.compile(r"[^\w\s']", flags=re.UNICODE)
    ws_re = re.compile(r"\s+")

    def normalise(text: str) -> list[str]:
        t = punct_re.sub(" ", text.lower())
        t = ws_re.sub(" ", t).strip()
        return t.split()

    def encode(words):
        tokens, ranges = [], []
        for i, w in enumerate(words):
            start = len(tokens)
            for c in w:
                if c in vocab:
                    tokens.append(vocab[c])
            end = len(tokens)
            ranges.append((start, end))
            if i < len(words) - 1 and sep_id is not None:
                tokens.append(sep_id)
        return tokens, ranges

    def ctc_path_to_token_frame_spans(path, labels, bid):
        t_len = len(path)
        i = 0
        spans = []
        for j, yj in enumerate(labels):
            while i < t_len and path[i] == bid:
                i += 1
            if i >= t_len:
                raise ValueError(f"CTC path ended early at label {j}/{len(labels)}")
            if path[i] != yj:
                raise ValueError(f"CTC path mismatch at label {j}: expected {yj}, got {path[i]}")
            start = i
            while i < t_len and path[i] == yj:
                i += 1
            spans.append((start, i))
        while i < t_len and path[i] == bid:
            i += 1
        if i != t_len:
            raise ValueError(f"CTC path has non-blank after last label (frame {i}, id={path[i]})")
        return spans

    @torch.inference_mode()
    def align_word_spans(wav_np, words):
        wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
        emission = torch.log_softmax(model(wav_t).logits, dim=-1)
        tokens, ranges = encode(words)
        if not tokens:
            return []
        targets = torch.tensor([tokens], dtype=torch.int32, device=device)
        alignments, _scores = forced_align(emission, targets, blank=blank_id)
        path = alignments[0].flatten().int().cpu().tolist()
        tok_spans = ctc_path_to_token_frame_spans(path, tokens, blank_id)
        t_frames = emission.size(1)
        sec_per_frame = len(wav_np) / TARGET_SR / t_frames
        out = []
        for ts, te in ranges:
            if te <= ts:
                continue
            out.append((tok_spans[ts][0] * sec_per_frame, tok_spans[te - 1][1] * sec_per_frame))
        return out

    def resample_16k(wav, sr):
        if sr == TARGET_SR:
            return wav, sr
        try:
            import torchaudio.functional as taf

            t = torch.from_numpy(wav).float()
            return taf.resample(t, sr, TARGET_SR).numpy(), TARGET_SR
        except Exception:
            from scipy.signal import resample_poly

            return resample_poly(wav, TARGET_SR, sr).astype(np.float32), TARGET_SR

    def reassembly_fields(clip_id, full_text, chunk_index, num_chunks):
        if not add_reassembly:
            return {}
        return {
            "clip_id": clip_id,
            "full_text": full_text,
            "chunk_index": chunk_index,
            "num_chunks": num_chunks,
        }

    rows, n_failed, n_chunked = [], 0, 0
    next_clip_id = 0
    for row in tqdm(ds, total=len(ds), desc=desc, unit="clip", dynamic_ncols=True):
        clip_id = next_clip_id
        next_clip_id += 1
        audio = row[AUDIO_COLUMN]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        dur = len(wav) / sr
        text = (row.get(TEXT_COLUMN) or "").strip()

        if dur <= MAX_AUDIO_SEC or not text:
            nr = dict(row)
            nr["chunk_index"] = 0
            nr.update(reassembly_fields(clip_id, text, 0, 1))
            rows.append(nr)
            continue

        try:
            wav, sr = resample_16k(wav, sr)
            norm_words = normalise(text)
            orig_words = text.split()
            if len(orig_words) != len(norm_words):
                orig_words = norm_words

            spans = align_word_spans(wav, norm_words)
            if not spans:
                raise ValueError("no words aligned")

            chunks = []
            chunk_start = spans[0][0]
            cur_idx = []
            for i, (s, e) in enumerate(spans):
                if e - chunk_start > max_chunk_sec and cur_idx:
                    chunks.append((chunk_start, spans[cur_idx[-1]][1], cur_idx[:]))
                    chunk_start = s
                    cur_idx = [i]
                else:
                    cur_idx.append(i)
            if cur_idx:
                chunks.append((chunk_start, spans[cur_idx[-1]][1], cur_idx[:]))

            n_chunked += 1
            n_chunks = len(chunks)
            for ci, (cs, ce, widx) in enumerate(chunks):
                ss = max(0, int(cs * sr))
                se = min(len(wav), int(ce * sr))
                seg = wav[ss:se]
                seg_text = " ".join(orig_words[j] for j in widx)
                nr = dict(row)
                nr[AUDIO_COLUMN] = {"array": seg, "sampling_rate": sr}
                nr[TEXT_COLUMN] = seg_text
                nr["chunk_index"] = ci
                nr.update(reassembly_fields(clip_id, text, ci, n_chunks))
                rows.append(nr)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  alignment failed for {dur:.1f}s clip "
                f"(source={row.get('source_dataset')}): {exc}"
            )
            n_failed += 1
            nr = dict(row)
            nr["chunk_index"] = 0
            nr.update(reassembly_fields(clip_id, text, 0, 1))
            rows.append(nr)

    n_in = len(ds)
    n_out = len(rows)
    print(
        f"  {n_in:,} clip(s) -> {n_out:,} row(s); "
        f"chunked {n_chunked} long clip(s); {n_failed} alignment failure(s)"
    )
    if add_reassembly and n_out != n_in:
        print(
            f"  (+{n_out - n_in:,} extra rows from splitting; "
            f"eval should reassemble by clip_id -> full_text)"
        )
    for r in rows:
        audio = r.get(AUDIO_COLUMN)
        if not audio or "array" not in audio:
            continue
        r[AUDIO_COLUMN] = {
            "array": np.asarray(audio["array"], dtype=np.float32),
            "sampling_rate": audio["sampling_rate"],
        }
    return Dataset.from_list(rows)
