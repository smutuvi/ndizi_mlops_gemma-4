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


@dataclass
class GemmaMixedCollator:
    """ASR rows (audio+text) or text-only chat rows (task=='chat'). Batch size must be 1."""

    proc: Any
    instruction: str = ASR_INSTRUCTION
    system_prompt: str = ""

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError("Set per_device_train_batch_size=1 for Gemma 4 mixed collator.")
        ex = batch[0]
        if str(ex.get("task") or "asr") == "chat":
            return self._chat(ex)
        return GemmaASRCollator(self.proc, instruction=self.instruction)(batch)

    def _chat(self, ex: dict[str, Any]):
        user = ex.get("prompt") or ex.get("instruction") or ""
        target = ex.get(TEXT_COLUMN) or ex.get("response") or ex.get("output") or ""
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": str(user)}]})
        messages.append({"role": "assistant", "content": str(target)})
        prompt = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if isinstance(prompt, list):
            prompt = prompt[0]
        inputs = self.proc(text=prompt, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        tids = self.proc.tokenizer(str(target), add_special_tokens=False)["input_ids"]
        cut = inputs["input_ids"].shape[1] - len(tids)
        labels[0, :cut] = -100
        pad_id = self.proc.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs
