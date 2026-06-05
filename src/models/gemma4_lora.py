# src/models/gemma4_lora.py — LoRA config that avoids Gemma4ClippableLinear in audio/vision towers.
#
# PEFT walks the full multimodal model when target_modules is a plain list like ["q_proj", ...].
# Audio/vision blocks also name layers q_proj/k_proj but wrap them in Gemma4ClippableLinear,
# which PEFT cannot inject into under 4-bit (Linear4bit inside the wrapper).
#
# Fix: scope target_modules to language_model paths only (regex). See:
# https://github.com/huggingface/peft/issues/3129
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)

# LM decoder only — excludes audio_tower / vision_tower ClippableLinear homonyms.
DEFAULT_GEMMA4_LM_LORA_TARGETS = (
    r"^(?=.*\.language_model\.)(?!.*\.(?:audio_tower|vision_tower)\.).*"
    r"\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

# Keep audio (and lm_head) in fp16/bf16 under 4-bit QLoRA — Gemma4ClippableLinear uses
# torch.finfo(weight.dtype), which fails on bitsandbytes packed uint8 storage.
# See: https://discuss.huggingface.co/t/issue-while-quantizing-gemma-4-e2b-e4b/176065
GEMMA4_BNB_SKIP_MODULES = [
    "lm_head",
    "audio_tower",
    "embed_audio",
    "model.audio_tower",
    "model.embed_audio",
]


def build_gemma4_bnb_config(*, compute_dtype=None):
    """4-bit QLoRA config with audio tower left unquantized."""
    import torch

    if compute_dtype is None:
        compute_dtype = torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=list(GEMMA4_BNB_SKIP_MODULES),
    )


def align_gemma4_multimodal_dtypes(model: Any, *, dtype=None) -> None:
    """
    Cast text/audio embedding paths to bf16 so Gemma4 masked_scatter dtypes match under QLoRA.

    With 4-bit LM + unquantized audio_tower, embed_tokens often stays float32 while audio
    features are bfloat16, which triggers:
      masked_scatter_: expected self and source to have same dtypes but got Float and BFloat16
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    keys = (
        "language_model.embed_tokens",
        "embed_audio",
        "audio_projector",
        "multi_modal_projector",
    )
    touched: list[str] = []
    for name, module in model.named_modules():
        if any(k in name for k in keys):
            module.to(dtype=dtype)
            touched.append(name)
    if touched:
        logger.info(
            "Aligned %d multimodal module(s) to %s for masked_scatter (e.g. %s)",
            len(touched),
            dtype,
            touched[0],
        )


def patch_masked_scatter_dtype_compat() -> None:
    """Align source dtype to destination before masked_scatter (Gemma 4 multimodal QLoRA)."""
    import torch

    if getattr(torch.Tensor, "_ndizi_masked_scatter_dtype_patch", False):
        return

    _orig = torch.Tensor.masked_scatter

    def masked_scatter(self, mask, source):
        if source.dtype != self.dtype:
            source = source.to(dtype=self.dtype, device=source.device)
        return _orig(self, mask, source)

    torch.Tensor.masked_scatter = masked_scatter  # type: ignore[method-assign]
    torch.Tensor._ndizi_masked_scatter_dtype_patch = True  # type: ignore[attr-defined]
    logger.info("Patched torch.Tensor.masked_scatter for dtype alignment (Gemma 4 QLoRA)")


def build_gemma4_lora_config(
    *,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: str | list[str] | None = None,
    modules_to_save: list[str] | None = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = DEFAULT_GEMMA4_LM_LORA_TARGETS
    if modules_to_save is None:
        modules_to_save = ["audio_projector", "multi_modal_projector"]
    return LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=target_modules,
        modules_to_save=list(modules_to_save),
        task_type="CAUSAL_LM",
    )


def count_lora_target_modules(model: Any, target_modules: str | list[str]) -> int:
    """Dry-run: how many named modules would match the LoRA target pattern."""
    if isinstance(target_modules, str):
        pat = re.compile(target_modules)
        return sum(1 for name, _ in model.named_modules() if pat.fullmatch(name))
    names = set(target_modules)
    return sum(1 for name, _ in model.named_modules() if name.split(".")[-1] in names)


def log_lora_target_preview(model: Any, target_modules: str | list[str], *, limit: int = 12) -> None:
    if isinstance(target_modules, str):
        pat = re.compile(target_modules)
        hits = [n for n, _ in model.named_modules() if pat.fullmatch(n)]
    else:
        suffixes = set(target_modules)
        hits = [n for n, _ in model.named_modules() if n.split(".")[-1] in suffixes]
    logger.info("LoRA target matches: %d modules (showing up to %d)", len(hits), limit)
    for n in hits[:limit]:
        logger.info("  %s", n)
    if len(hits) > limit:
        logger.info("  ...")


def infer_lm_lora_regex_from_model(model: Any) -> str:
    """Fallback regex when DEFAULT_GEMMA4_LM_LORA_TARGETS matches nothing on this transformers build."""
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for name, _ in model.named_modules():
        if not name.endswith(suffixes):
            continue
        if "audio_tower" in name or "vision_tower" in name:
            continue
        logger.info("Inferred LoRA targets using decoder layer pattern (sample: %s)", name)
        return (
            r"^(?!.*(?:audio_tower|vision_tower)).*\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))$"
        )
    raise RuntimeError("Could not infer language-model LoRA targets from model.named_modules()")


def apply_gemma4_lora(model: Any, lora_config: LoraConfig, *, debug_targets: bool = False) -> Any:
    targets = lora_config.target_modules
    n = count_lora_target_modules(model, targets)
    if n == 0 and isinstance(targets, str):
        inferred = infer_lm_lora_regex_from_model(model)
        lora_config = build_gemma4_lora_config(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=inferred,
            modules_to_save=lora_config.modules_to_save,
        )
        targets = lora_config.target_modules
        n = count_lora_target_modules(model, targets)
    if debug_targets:
        log_lora_target_preview(model, targets)
    if n == 0:
        raise RuntimeError(
            "LoRA target regex matched 0 modules. Pass --debug-lora-targets or --lora-target-modules."
        )
    if debug_targets:
        logger.info("LoRA will attach to %d modules", n)
    try:
        return get_peft_model(model, lora_config)
    except ValueError as e:
        if "Gemma4ClippableLinear" not in str(e):
            raise
        raise RuntimeError(
            "PEFT failed on Gemma4ClippableLinear. Upgrade peft (>=0.15 recommended) or use "
            "scripts/train_gemma4.py --no-4bit for bf16 LoRA. "
            f"Original error: {e}"
        ) from e


def patch_clippable_linear_for_peft() -> None:
    """
    Last-resort monkey-patch so ClippableLinear passes isinstance(..., nn.Linear).

    Not used by default with 4-bit QLoRA (can interfere with quantization). Prefer LM-only regex.
    """
    try:
        import transformers.models.gemma4.modeling_gemma4 as modeling_gemma4
    except ImportError as e:
        raise RuntimeError("transformers Gemma 4 modeling not available") from e

    if getattr(modeling_gemma4.Gemma4ClippableLinear, "_ndizi_peft_patched", False):
        return

    _Orig = modeling_gemma4.Gemma4ClippableLinear

    import torch

    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features, *args, **kwargs):
            nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, input):
            if self.use_clipped_linears:
                input = torch.clamp(input, min=self.input_min, max=self.input_max)
            out = nn.Linear.forward(self, input)
            if self.use_clipped_linears:
                out = torch.clamp(out, min=self.output_min, max=self.output_max)
            return out

    PatchedClippableLinear._ndizi_peft_patched = True  # type: ignore[attr-defined]
    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
    logger.warning("Applied Gemma4ClippableLinear PEFT monkey-patch (use only if regex LoRA fails)")


# ── asr_safe: projector-only training (no LM LoRA) ────────────────────────────

PROJECTOR_MODULE_KEYS = ("audio_projector", "multi_modal_projector")


def freeze_lm_decoder(model: Any) -> Any:
    """Freeze all parameters except audio_projector and multi_modal_projector.

    Used for asr_safe training mode — only the multimodal projection layers are
    updated, leaving the LM decoder weights completely unchanged.
    """
    for name, param in model.named_parameters():
        if any(k in name for k in PROJECTOR_MODULE_KEYS):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "[asr_safe] Trainable params: %s / %s (%.2f%%) — projectors only",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )
    return model


def save_projector_checkpoint(model: Any, out_dir: Path | str) -> None:
    """Save only projector weights + a mode marker. Compatible with load_projector_checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.cpu().contiguous()
        for k, v in model.state_dict().items()
        if any(key in k for key in PROJECTOR_MODULE_KEYS)
    }
    save_file(state, out_dir / "projector_weights.safetensors")
    (out_dir / "training_mode.json").write_text(
        json.dumps({"training_mode": "asr_safe", "saved_modules": list(PROJECTOR_MODULE_KEYS)}),
        encoding="utf-8",
    )
    logger.info("Saved projector-only checkpoint (%d tensors) to %s", len(state), out_dir)


def is_projector_only_checkpoint(adapter_dir: Path | str) -> bool:
    return (Path(adapter_dir) / "training_mode.json").exists()


def load_projector_checkpoint(model: Any, adapter_dir: Path | str) -> Any:
    """Overlay projector weights from an asr_safe checkpoint onto a base model."""
    state = load_file(str(Path(adapter_dir) / "projector_weights.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        logger.warning("Unexpected keys in projector checkpoint: %s", unexpected[:5])
    logger.info(
        "Loaded projector weights (%d tensors); %d keys not in checkpoint",
        len(state), len(missing),
    )
    return model


# ── asr_moderate: tail-LoRA on last N decoder layers + full projector save ────

def _count_decoder_layers(model: Any) -> int:
    max_idx = -1
    for name, _ in model.named_modules():
        m = re.search(r"language_model\.model\.layers\.(\d+)\.", name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx < 0:
        raise RuntimeError("Could not determine decoder layer count from model.named_modules()")
    return max_idx + 1


def build_asr_moderate_lora_config(
    model: Any,
    *,
    num_tail_layers: int = 6,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """LoRA on the last num_tail_layers decoder layers only, plus full projector saves."""
    total = _count_decoder_layers(model)
    first_tail = max(0, total - num_tail_layers)
    tail_indices = "|".join(str(i) for i in range(first_tail, total))
    target_regex = (
        r"^(?=.*\.language_model\.)"
        rf"(?=.*\.layers\.({tail_indices})\.).*"
        r"\.(?:self_attn|mlp)\."
        r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )
    logger.info(
        "[asr_moderate] Tail LoRA: layers %d–%d of %d (r=%d, alpha=%d)",
        first_tail, total - 1, total, r, lora_alpha,
    )
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_regex,
        modules_to_save=list(PROJECTOR_MODULE_KEYS),
        task_type="CAUSAL_LM",
    )
