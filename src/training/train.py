# src/training/train.py — QLoRA fine-tune Gemma 4 for Swahili ASR (adapter-first).
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import torch
from datasets import interleave_datasets, load_from_disk
from jiwer import wer
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments

from src.models.gemma4_lora import (
    align_gemma4_multimodal_dtypes,
    apply_gemma4_lora,
    build_asr_moderate_lora_config,
    build_gemma4_bnb_config,
    build_gemma4_lora_config,
    ensure_linear4bit_quant_state,
    freeze_lm_decoder,
    patch_clippable_linear_for_peft,
    patch_masked_scatter_dtype_compat,
    qlora_device_map,
    repair_kv_shared_dummy_projections,
    rewrite_adapter_config_for_kv_shared,
    save_projector_checkpoint,
)
from src.training.collator import GemmaASRCollator, GemmaMixedCollator
from src.training.gemma_trainer import GemmaASRTrainer
from src.training.retention import maybe_load_retention_replay_train
from src.utils.constants import (
    AUDIO_COLUMN,
    SUNFLOWER_SYSTEM_PROMPT,
    TARGET_SR,
    resolve_asr_prompt,
)
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
    # Allow --output-dir override; fall back to CHECKPOINT_DIR from paths.py
    _out_dir = getattr(cli_args, "output_dir", None)
    checkpoint_dir = Path(_out_dir) if _out_dir else CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dsd = load_from_disk(str(Path(getattr(cli_args, "prepared_dir", None) or PREPARED_LOCAL)))
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
            [train_ds, retention_train],
            probabilities=[p_dom, p_ret],
            seed=42,
            stopping_strategy="all_exhausted",
        )

    chat_jsonl = getattr(cli_args, "chat_jsonl", None)
    chat_ratio = float(getattr(cli_args, "chat_ratio", 0.0) or 0.0)
    if getattr(cli_args, "training_mode", "asr_max") == "asr_safe" and chat_ratio > 0:
        print("[train] asr_safe: skipping chat mix (LM is frozen; dummy-audio chat rows would poison audio_tower)")
        chat_jsonl = None
        chat_ratio = 0.0
    if chat_jsonl and chat_ratio > 0:
        import numpy as np
        from datasets import Dataset as HfDataset

        path = Path(chat_jsonl)
        if not path.is_file():
            raise SystemExit(f"--chat-jsonl not found: {path}")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt") or rec.get("instruction") or rec.get("user") or ""
            response = rec.get("response") or rec.get("output") or rec.get("text") or ""
            if not prompt or not response:
                continue
            rows.append((prompt, response))
        if not rows:
            raise SystemExit(f"--chat-jsonl has no valid prompt/response rows: {path}")
        # interleave_datasets requires matching features — pad chat rows to the ASR schema.
        if "prompt" not in train_ds.column_names:
            train_ds = train_ds.add_column("prompt", [""] * len(train_ds))
        dummy_audio = {
            "array": np.zeros(int(0.16 * TARGET_SR), dtype=np.float32),
            "sampling_rate": TARGET_SR,
        }
        proto = train_ds[0]
        chat_rows = []
        for prompt, response in rows:
            row = {k: proto[k] for k in train_ds.column_names}
            row["prompt"] = prompt
            row["text"] = response
            row["task"] = "chat"
            row["source_dataset"] = "chat_jsonl"
            if AUDIO_COLUMN in row:
                row[AUDIO_COLUMN] = dummy_audio
            chat_rows.append(row)
        chat_ds = HfDataset.from_list(chat_rows).cast(train_ds.features)
        n_asr = len(train_ds)
        p_chat = min(max(chat_ratio, 0.0), 0.5)
        p_asr = 1.0 - p_chat
        print(
            f"[train] chat mix enabled: chat_ratio={p_chat:.3f} (asr={p_asr:.3f}, "
            f"n_asr={n_asr}, n_chat={len(chat_ds)}; cycling chat until ASR is exhausted)"
        )
        train_ds = interleave_datasets(
            [train_ds, chat_ds],
            probabilities=[p_asr, p_chat],
            seed=42,
            stopping_strategy="all_exhausted",
        )

    training_mode = getattr(cli_args, "training_mode", "asr_max")
    prompt_style = getattr(cli_args, "asr_prompt", None)
    if not prompt_style:
        prompt_style = "short" if getattr(cli_args, "short_instruction", False) else "full"
    asr_instruction = resolve_asr_prompt(prompt_style)
    print(f"[train] Training mode : {training_mode}")
    print(f"[train] ASR prompt    : {prompt_style} ({asr_instruction[:60]}{'…' if len(asr_instruction) > 60 else ''})")
    n_train = len(train_ds)
    grad_accum = int(getattr(cli_args, "grad_accum", 16))
    epochs = float(getattr(cli_args, "epochs", 2.0))
    steps_est = max(1, int((n_train / max(grad_accum, 1)) * epochs))
    print(f"[train] Train rows: {n_train:,}  (~{steps_est} optimizer steps at batch=1, accum={grad_accum}, epochs={epochs:g})")

    processor = AutoProcessor.from_pretrained(rt.base_model_id, padding_side="left")

    use_4bit = not bool(getattr(cli_args, "no_4bit", False))
    if getattr(cli_args, "peft_clippable_patch", False):
        patch_clippable_linear_for_peft()

    if training_mode == "asr_safe":
        # Audio-path SFT: no quantization, no LoRA — embed_audio + optional audio_tower.
        print("[train] asr_safe: loading bf16 model, freezing LM decoder, training audio path")
        model = AutoModelForMultimodalLM.from_pretrained(
            rt.base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        n_kv = repair_kv_shared_dummy_projections(model)
        if n_kv:
            print(f"[train] Repaired {n_kv} dummy KV-shared k/v projections")
        model = freeze_lm_decoder(
            model,
            include_audio_tower=bool(getattr(cli_args, "unfreeze_audio_tower", False)),
            audio_tower_last_layers=int(getattr(cli_args, "audio_tower_last_layers", 4)),
        )
        model.train()
    else:
        # asr_moderate or asr_max: 4-bit QLoRA
        if use_4bit:
            patch_masked_scatter_dtype_compat()
        load_kw: dict = dict(attn_implementation="sdpa")
        if use_4bit:
            # Do not pass dtype= with BitsAndBytesConfig — it unpacks Params4bit
            # ([out, in] instead of packed [out, 1]) and the first k_proj forward
            # raises assert module.weight.shape[1] == 1.
            load_kw["quantization_config"] = build_gemma4_bnb_config()
            load_kw["device_map"] = qlora_device_map()
            print(
                f"[train] Loading base model with 4-bit QLoRA "
                f"(device_map={load_kw['device_map']}; audio_tower skipped; bf16 compute)"
            )
        else:
            load_kw["dtype"] = torch.bfloat16
            load_kw["device_map"] = "auto"
            print("[train] Loading base model in bf16 (no 4-bit)")
        model = AutoModelForMultimodalLM.from_pretrained(rt.base_model_id, **load_kw)
        if getattr(model.config, "use_cache", None):
            model.config.use_cache = False
        n_kv = repair_kv_shared_dummy_projections(model)
        if n_kv:
            print(
                f"[train] Repaired {n_kv} dummy KV-shared k/v projections "
                "(Sunflower checkpoint omits them; 4-bit init was crashing k_proj)"
            )
        if use_4bit:
            n_qs = ensure_linear4bit_quant_state(model)
            if n_qs:
                print(f"[train] Initialized 4-bit quant_state on {n_qs} Linear4bit modules")
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            align_gemma4_multimodal_dtypes(model)
            n_qs = ensure_linear4bit_quant_state(model)
            if n_qs:
                print(f"[train] Re-initialized 4-bit quant_state after kbit prep ({n_qs} modules)")

        if training_mode == "asr_moderate":
            num_tail = int(getattr(cli_args, "tail_lora_layers", 6))
            rank = int(getattr(cli_args, "tail_lora_rank", 8))
            lora = build_asr_moderate_lora_config(model, num_tail_layers=num_tail, r=rank, lora_alpha=rank * 2)
        else:
            # asr_max — existing full-decoder LoRA behaviour
            lora_target = getattr(cli_args, "lora_target_modules", None)
            lora = build_gemma4_lora_config(model, target_modules=lora_target)

        model = apply_gemma4_lora(
            model,
            lora,
            debug_targets=bool(getattr(cli_args, "debug_lora_targets", False)),
            include_audio_tower=bool(getattr(cli_args, "unfreeze_audio_tower", False)),
            audio_tower_last_layers=int(getattr(cli_args, "audio_tower_last_layers", 4)),
        )
        model.print_trainable_parameters()
        if use_4bit:
            n_qs = ensure_linear4bit_quant_state(model)
            if n_qs:
                print(f"[train] Re-initialized 4-bit quant_state after LoRA ({n_qs} modules)")

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
        output_dir=str(checkpoint_dir),
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

    collator = GemmaASRCollator(processor, instruction=asr_instruction)
    if chat_jsonl and chat_ratio > 0:
        collator = GemmaMixedCollator(
            processor,
            instruction=asr_instruction,
            system_prompt=str(getattr(cli_args, "system_prompt", None) or SUNFLOWER_SYSTEM_PROMPT),
        )

    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
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

    out_dir = checkpoint_dir / "best"
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt:
        print(f"[train] Best checkpoint: {best_ckpt} (eval WER={getattr(trainer.state, 'best_metric', None)})")
    else:
        print(f"[train] No best checkpoint from eval; saving latest weights to {out_dir}")

    if training_mode == "asr_safe":
        # Save only the projector weights — avoids writing a 5GB full model checkpoint.
        save_projector_checkpoint(trainer.model, out_dir, training_mode="asr_safe")
        processor.save_pretrained(str(out_dir))
        print(f"[asr_safe] Saved projector-only checkpoint to {out_dir}")
    else:
        trainer.save_model(str(out_dir))
        processor.save_pretrained(str(out_dir))
        if rewrite_adapter_config_for_kv_shared(out_dir, trainer.model):
            print("[train] Patched adapter_config.json for Gemma 4 KV-shared layers")
        # embed_audio is not PEFT modules_to_save (kwargs-only forward). Persist it beside LoRA.
        save_projector_checkpoint(trainer.model, out_dir, training_mode=training_mode)
        print("Saved LoRA adapter + embed_audio to", out_dir)
