# src/utils/runtime.py — resolved base model + Hub output repo for the current run.
from __future__ import annotations

from dataclasses import dataclass

MODEL_ALIASES = {
    "E2B": "google/gemma-4-E2B-it",
    "E4B": "google/gemma-4-E4B-it",
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
}


@dataclass
class GemmaRuntime:
    base_model_id: str = "google/gemma-4-E2B-it"
    output_model_repo: str = "smutuvi/gemma-4-e2b-sw-asr-ndizi"

    @property
    def merged_model_repo(self) -> str:
        return f"{self.output_model_repo}-merged"


_runtime = GemmaRuntime()


def get_runtime() -> GemmaRuntime:
    return _runtime


def resolve_model_id(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def apply_model_choice(name: str) -> GemmaRuntime:
    global _runtime
    base = resolve_model_id(name)
    short = base.split("/")[-1].replace("-it", "").lower()
    _runtime = GemmaRuntime(
        base_model_id=base,
        output_model_repo=f"smutuvi/{short}-sw-asr-ndizi",
    )
    print(f"[config] BASE_MODEL_ID     = {_runtime.base_model_id}")
    print(f"[config] OUTPUT_MODEL_REPO = {_runtime.output_model_repo}")
    print(f"[config] MERGED_MODEL_REPO  = {_runtime.merged_model_repo}")
    return _runtime
