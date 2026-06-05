# src/training/collator.py — per-example Gemma 4 ASR collator (batch size must be 1).
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.inference.gemma_inputs import gemma_build_inputs, prepare_audio_for_gemma
from src.utils.constants import ASR_INSTRUCTION, AUDIO_COLUMN, TEXT_COLUMN


@dataclass
class GemmaASRCollator:
    """Gemma 4 audio-token accounting requires one audio per batch."""

    proc: Any
    instruction: str = ASR_INSTRUCTION

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError(
                "Set per_device_train_batch_size=1 for Gemma 4 audio; "
                "batched audio triggers a token/feature mismatch."
            )
        ex = batch[0]
        target = ex[TEXT_COLUMN]
        wave = prepare_audio_for_gemma(ex[AUDIO_COLUMN])
        inputs = gemma_build_inputs(
            self.proc,
            wave,
            self.instruction,
            add_generation_prompt=False,
            assistant_text=target,
        )
        labels = inputs["input_ids"].clone()
        tids = self.proc.tokenizer(target, add_special_tokens=False)["input_ids"]
        cut = inputs["input_ids"].shape[1] - len(tids)
        labels[0, :cut] = -100
        labels[labels == self.proc.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
