# src/data/splits.py — parse Hub dataset id specs (repo or repo:split).
from __future__ import annotations


def split_spec_list(raw: list[str] | None, *, default_split: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw or []:
        s = str(item).strip()
        if not s:
            continue
        if ":" in s:
            did, sp = s.split(":", 1)
            out.append((did.strip(), sp.strip()))
        else:
            out.append((s, default_split))
    return out
