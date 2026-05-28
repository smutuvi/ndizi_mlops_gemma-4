# src/training/train.py — QLoRA fine-tune Gemma 4 for Swahili ASR (adapter-first).
from __future__ import annotations

import inspect
import os

import torch
from datasets import interleave_datasets, load_from_disk
from jiwer import wer
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

from src.models.gemma4_lora import (
    apply_gemma4_lora,
    build_gemma4_lora_config,
    patch_clippable_linear_for_peft,
    patch_gemma4_audio_ffn_finfo_for_kbit,
)
from src.training.collator import GemmaASRCollator
from src.training.gemma_trainer import GemmaASRTrainer
from src.training.retention import maybe_load_retention_replay_train
from src.utils.paths import CHECKPOINT_DIR, PREPARED_LOCAL
from src.utils.runtime import get_runtime


def _training_arguments(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _trainer_class():
    return GemmaASRTrainer


def _trainer(**kwargs):
    sig = inspect.signature(Trainer.__init__)
    cls = _trainer_class()
    return cls(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _strip_label_positions(pred_row, label_row):
    kept_p, kept_l = [], []
    for p, l in zip(pred_row, label_row):
        if int(l) == -100:
            continue
        kept_p.append(int(p))
        kept_l.append(int(l))
    return kept_p, kept_l


def run_train(cli_args) -> None:
    rt = get_runtime()
    dsd = load_from_disk(str(PREPARED_LOCAL))
    train_ds = dsd["train"]

    eval_max = int(getattr(cli_args, "eval_max_samples", 64))
    skip_eval = bool(getattr(cli_args, "no_train_eval", False))
    if skip_eval:
        eval_ds = None
        print("[train] Mid-training eval disabled (--no-train-eval); checkpoints saved by save_steps only.")
    else:
        n_eval = min(eval_max, len(dsd["validation"]))
        eval_ds = dsd["validation"].select(range(n_eval))
        print(f"[train] Eval subset: {n_eval} validation rows (cap --eval-max-samples={eval_max})")

    retention_train, replay_ratio = maybe_load_retention_replay_train(cli_args)
    if retention_train is not None and replay_ratio > 0:
        p_ret = min(max(replay_ratio, 0.0), 0.5)
        p_dom = 1.0 - p_ret
        print(f"[train] replay mix enabled: retention_ratio={p_ret:.3f} (domain={p_dom:.3f})")
        train_ds = interleave_datasets(
            [train_ds, retention_train], probabilities=[p_dom, p_ret], seed=42
        )

    processor = AutoProcessor.from_pretrained(rt.base_model_id, padding_side="left")

    use_4bit = not bool(getattr(cli_args, "no_4bit", False))
    if use_4bit:
        patch_gemma4_audio_ffn_finfo_for_kbit()
    if getattr(cli_args, "peft_clippable_patch", False):
        patch_clippable_linear_for_peft()

    load_kw: dict = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kw["quantization_config"] = bnb
        print("[train] Loading base model with 4-bit QLoRA (BitsAndBytes nf4)")
    else:
        print("[train] Loading base model in bf16 (no 4-bit); needs more VRAM but avoids ClippableLinear LoRA issues")

    model = AutoModelForMultimodalLM.from_pretrained(rt.base_model_id, **load_kw)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_target = getattr(cli_args, "lora_target_modules", None)
    lora = build_gemma4_lora_config(target_modules=lora_target)
    model = apply_gemma4_lora(model, lora, debug_targets=bool(getattr(cli_args, "debug_lora_targets", False)))
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        import numpy as np

        if hasattr(preds, "ndim") and preds.ndim == 3:
            pred_ids = np.argmax(preds, axis=-1).astype(np.int64)
        else:
            pred_ids = preds

        pred_rows, label_rows = [], []
        for p_row, l_row in zip(pred_ids, labels):
            kp, kl = _strip_label_positions(p_row, l_row)
            pred_rows.append(kp)
            label_rows.append(kl)

        return {
            "wer": wer(
                processor.batch_decode(label_rows, skip_special_tokens=True),
                processor.batch_decode(pred_rows, skip_special_tokens=True),
            )
        }

    eval_steps = int(getattr(cli_args, "eval_steps", 500))
    save_steps = int(getattr(cli_args, "save_steps", 500))
    ta_sig = inspect.signature(TrainingArguments.__init__)
    strategy_kw: dict = {}
    if skip_eval:
        if "eval_strategy" in ta_sig.parameters:
            strategy_kw["eval_strategy"] = "no"
        else:
            strategy_kw["evaluation_strategy"] = "no"
    else:
        if "eval_strategy" in ta_sig.parameters:
            strategy_kw["eval_strategy"] = "steps"
        else:
            strategy_kw["evaluation_strategy"] = "steps"
        strategy_kw["eval_steps"] = eval_steps
    if "save_strategy" in ta_sig.parameters:
        strategy_kw["save_strategy"] = "steps"

    ta_kw = dict(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(getattr(cli_args, "grad_accum", 16)),
        num_train_epochs=float(getattr(cli_args, "epochs", 2.0)),
        learning_rate=float(getattr(cli_args, "lr", 2e-4)),
        lr_scheduler_type=str(getattr(cli_args, "lr_scheduler", "cosine")),
        warmup_ratio=float(getattr(cli_args, "warmup_ratio", 0.03)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_steps=save_steps,
        save_total_limit=int(getattr(cli_args, "save_total_limit", 3)),
        load_best_model_at_end=not skip_eval,
        greater_is_better=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    if not skip_eval:
        ta_kw["metric_for_best_model"] = "wer"
    training_args = _training_arguments(**{**ta_kw, **strategy_kw})

    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=GemmaASRCollator(processor),
    )
    if eval_ds is not None:
        trainer_kw["eval_dataset"] = eval_ds
        trainer_kw["compute_metrics"] = compute_metrics

    tr_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in tr_sig.parameters:
        trainer_kw["processing_class"] = processor
    elif "tokenizer" in tr_sig.parameters:
        trainer_kw["tokenizer"] = processor

    trainer = _trainer(**trainer_kw)
    trainer.train()

    out_dir = CHECKPOINT_DIR / "best"
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt:
        print(f"[train] Best checkpoint: {best_ckpt} (eval WER={getattr(trainer.state, 'best_metric', None)})")
    else:
        print(f"[train] No best checkpoint from eval; saving latest weights to {out_dir}")

    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print("Saved LoRA adapter to", out_dir)
