# src/models/gemma4_lora.py — LoRA config that avoids Gemma4ClippableLinear in audio/vision towers.
#
# PEFT walks the full multimodal model when target_modules is a plain list like ["q_proj", ...].
# Audio/vision blocks also name layers q_proj/k_proj but wrap them in Gemma4ClippableLinear,
# which PEFT cannot inject into under 4-bit (Linear4bit inside the wrapper).
#
# Fix: scope target_modules to language_model paths only (regex). See:
# https://github.com/huggingface/peft/issues/3129
from __future__ import annotations

import logging
import re
from typing import Any

import torch.nn as nn
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# LM decoder only — excludes audio_tower / vision_tower ClippableLinear homonyms.
DEFAULT_GEMMA4_LM_LORA_TARGETS = (
    r"^(?=.*\.language_model\.)(?!.*\.(?:audio_tower|vision_tower)\.).*"
    r"\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


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


def patch_gemma4_audio_ffn_finfo_for_kbit() -> None:
    """
    4-bit weights use non-float storage dtypes; Gemma4AudioFeedForward.forward calls
    torch.finfo(self.ffw_layer_1.linear.weight.dtype) and crashes under QLoRA.
    """
    try:
        import torch
        import transformers.models.gemma4.modeling_gemma4 as modeling_gemma4
    except ImportError as e:
        raise RuntimeError("transformers Gemma 4 modeling not available") from e

    cls = modeling_gemma4.Gemma4AudioFeedForward
    if getattr(cls, "_ndizi_kbit_finfo_patch", False):
        return

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        wdtype = self.ffw_layer_1.linear.weight.dtype
        if getattr(wdtype, "is_floating_point", False):
            finfo_dtype = wdtype
        elif hidden_states.is_floating_point():
            finfo_dtype = hidden_states.dtype
        else:
            finfo_dtype = torch.bfloat16
        gradient_clipping = min(self.gradient_clipping, torch.finfo(finfo_dtype).max)

        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states *= self.post_layer_scale
        hidden_states += residual

        return hidden_states

    cls.forward = forward  # type: ignore[method-assign]
    cls._ndizi_kbit_finfo_patch = True  # type: ignore[attr-defined]
    logger.info("Patched Gemma4AudioFeedForward.forward for 4-bit QLoRA (safe torch.finfo)")


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
