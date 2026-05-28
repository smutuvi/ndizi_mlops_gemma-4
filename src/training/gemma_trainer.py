# src/training/gemma_trainer.py — Trainer that avoids OOM from storing full eval logits.
from __future__ import annotations

import torch
from transformers import Trainer


class GemmaASRTrainer(Trainer):
    """
    Gemma 4 eval would OOM if Trainer concatenates full logits (batch x seq x vocab).

    During eval we return argmax token ids (int64 on CPU) instead of logits so
    compute_metrics can decode without a multi‑GB logits buffer.
    """

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            cm = (
                self.compute_loss_context_manager()
                if hasattr(self, "compute_loss_context_manager")
                else self.autocast_smart_context_manager()
            )
            with cm:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if not has_labels or loss is None:
            return (loss, None, None)

        logits = outputs.logits
        # [batch, seq] int64 — ~4 bytes/token vs vocab*4 bytes for float logits.
        pred_ids = logits.argmax(dim=-1).detach().cpu()
        labels = inputs["labels"].detach().cpu()

        del logits, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (loss.detach(), pred_ids, labels)
