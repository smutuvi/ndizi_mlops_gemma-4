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
# Legacy regex (over-declares k_proj/v_proj on KV-shared layers). Prefer build_kv_aware_lm_lora_regex().
DEFAULT_GEMMA4_LM_LORA_TARGETS = (
    r"^(?=.*\.language_model\.)(?!.*\.(?:audio_tower|vision_tower)\.).*"
    r"\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

_LM_LORA_PREFIX = (
    r"^(?=.*\.language_model\.)(?!.*\.(?:audio_tower|vision_tower)\.).*"
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


def qlora_device_map():
    """Pin QLoRA to one CUDA device.

    ``device_map="auto"`` plus ``dtype=`` unpacks Linear4bit weights, which then fail on the
    first forward with ``FP4 quantization state not initialized`` /
    ``assert module.weight.shape[1] == 1`` (often on ``k_proj``).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA requires CUDA. Pass --no-4bit to train in bf16 on CPU.")
    return {"": int(torch.cuda.current_device())}


def _is_bnb_4bit_linear(module: Any) -> bool:
    name = type(module).__name__
    if name in ("Linear4bit", "LinearFP4", "LinearNF4"):
        return True
    mod = module.__class__.__module__
    return mod.startswith("bitsandbytes.nn") and "Linear" in name


def ensure_linear4bit_quant_state(model: Any) -> int:
    """Call ``.to(cuda)`` on packed 4-bit linears whose ``quant_state`` was never built.

    Unpacked weights (``shape[1] != 1``) are dummy KV-shared ``k_proj``/``v_proj``
    that Sunflower does not store — ``.to(cuda)`` cannot recover those. See
    ``repair_kv_shared_dummy_projections``.
    """
    import torch

    if not torch.cuda.is_available():
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    n = 0
    skipped_unpacked = 0
    for module in model.modules():
        if not _is_bnb_4bit_linear(module):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "quant_state", None) is not None:
            continue
        if getattr(weight, "ndim", 0) == 2 and int(weight.shape[1]) != 1:
            skipped_unpacked += 1
            continue
        module.to(device)
        n += 1
    if skipped_unpacked:
        logger.info(
            "Left %d unpacked Linear4bit modules for KV-shared repair (not packed NF4)",
            skipped_unpacked,
        )
    if n:
        logger.info("Initialized bitsandbytes quant_state on %d Linear4bit modules", n)
    return n


def _iter_text_self_attn(model: Any):
    for name, module in model.named_modules():
        if "audio_tower" in name or "vision_tower" in name:
            continue
        if getattr(module, "q_proj", None) is None:
            continue
        if getattr(module, "layer_idx", None) is None:
            continue
        yield name, module


def _linear_weight_bf16(src: Any, device):
    """Materialize an nn.Linear in bf16 from a (possibly 4-bit) donor projection."""
    import torch
    import torch.nn as nn

    weight = src.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        import bitsandbytes.functional as bnb_f

        data = bnb_f.dequantize_4bit(weight, quant_state)
    else:
        data = weight.detach().float()
    if data.ndim != 2 or int(data.shape[1]) == 1:
        raise ValueError(f"Cannot convert projection with shape {tuple(data.shape)} to nn.Linear")
    out_f, in_f = int(data.shape[0]), int(data.shape[1])
    has_bias = getattr(src, "bias", None) is not None
    lin = nn.Linear(in_f, out_f, bias=has_bias)
    lin.weight.data.copy_(data.to(dtype=torch.bfloat16))
    if has_bias:
        lin.bias.data.copy_(src.bias.detach().to(dtype=torch.bfloat16))
    lin.requires_grad_(False)
    return lin.to(device=device, dtype=torch.bfloat16)


def repair_kv_shared_dummy_projections(model: Any) -> int:
    """Replace randomly-initialized KV-shared k/v projections with the last real donor.

    Sunflower omits ``k_proj``/``v_proj``/``k_norm`` on trailing KV-shared layers.
    Some transformers builds still construct them; bitsandbytes then 4-bit-inits
    unpacked tensors and crashes on the first ``k_proj`` forward
    (``assert module.weight.shape[1] == 1``). Newer modeling skips those modules
    when ``is_kv_shared_layer`` is set; older modeling still calls ``k_proj``, so
    we copy the last non-shared layer of the same attention type as bf16.
    """
    import copy

    num_kv = _get_num_kv_shared_layers(model)
    total = _count_decoder_layers(model)
    first_shared = total - num_kv if num_kv > 0 else None

    dummy_idxs: list[int] = []
    for _name, attn in _iter_text_self_attn(model):
        kp = getattr(attn, "k_proj", None)
        weight = getattr(kp, "weight", None) if kp is not None else None
        if (
            weight is not None
            and getattr(weight, "quant_state", None) is None
            and getattr(weight, "ndim", 0) == 2
            and int(weight.shape[1]) != 1
        ):
            dummy_idxs.append(int(attn.layer_idx))
    if dummy_idxs:
        inferred = min(dummy_idxs)
        if first_shared is None or inferred < first_shared:
            first_shared = inferred

    if first_shared is None:
        return 0
    device = next(p.device for p in model.parameters() if p.device.type != "meta")

    donors: dict[str, Any] = {}
    n_replaced = 0
    for _name, attn in sorted(_iter_text_self_attn(model), key=lambda kv: int(kv[1].layer_idx)):
        idx = int(attn.layer_idx)
        layer_type = str(getattr(attn, "layer_type", None) or "default")
        if idx < first_shared:
            donors[layer_type] = attn
            continue
        attn.is_kv_shared_layer = True
        donor = donors.get(layer_type) or (next(iter(donors.values())) if donors else None)
        if donor is None:
            logger.warning("No KV donor for shared attention layer %d", idx)
            continue
        if getattr(attn, "k_proj", None) is not None and getattr(donor, "k_proj", None) is not None:
            attn.k_proj = _linear_weight_bf16(donor.k_proj, device)
            n_replaced += 1
        if getattr(attn, "v_proj", None) is not None and getattr(donor, "v_proj", None) is not None:
            attn.v_proj = _linear_weight_bf16(donor.v_proj, device)
            n_replaced += 1
        if getattr(donor, "k_norm", None) is not None and getattr(attn, "k_norm", None) is not None:
            attn.k_norm = copy.deepcopy(donor.k_norm).to(device)
        if getattr(donor, "v_norm", None) is not None and getattr(attn, "v_norm", None) is not None:
            attn.v_norm = copy.deepcopy(donor.v_norm).to(device)

    if n_replaced:
        logger.info(
            "Repaired %d KV-shared k/v projections (layers %d–%d of %d) from last donor",
            n_replaced,
            first_shared,
            total - 1,
            total,
        )
    return n_replaced


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
        if not any(k in name for k in keys):
            continue
        # ``module.to(dtype=)`` unpacks Linear4bit children and drops quant_state.
        if any(_is_bnb_4bit_linear(child) for child in module.modules()):
            continue
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


def _get_num_kv_shared_layers(model: Any) -> int:
    """Gemma 4 E2B/E4B: last N decoder layers share K/V and omit k_proj/v_proj."""
    config = getattr(model, "config", None)
    if config is None:
        return 0
    candidates = [config, getattr(config, "text_config", None)]
    for cfg in candidates:
        if cfg is None:
            continue
        n = getattr(cfg, "num_kv_shared_layers", None)
        if n is not None:
            return int(n)
    return 0


def _last_layer_with_kv_proj(model: Any) -> int | None:
    total = _count_decoder_layers(model)
    num_kv = _get_num_kv_shared_layers(model)
    if num_kv <= 0:
        return total - 1
    return total - num_kv - 1


def build_kv_aware_lm_lora_regex(model: Any, *, layer_filter: str | None = None) -> str:
    """
    LoRA regex aligned with Gemma 4 module layout.

    q_proj/o_proj and MLP LoRA apply on all target layers; k_proj/v_proj only on layers
    that actually exist (non-KV-shared). Tail-only configs on KV-shared layers get no k/v.
    """
    total = _count_decoder_layers(model)
    last_kv = _last_layer_with_kv_proj(model)

    if layer_filter:
        allowed = {int(x) for x in layer_filter.split("|")}
    else:
        allowed = set(range(total))

    all_layers = sorted(allowed)
    kv_layers = sorted(i for i in all_layers if last_kv is not None and i <= last_kv)
    all_indices = "|".join(str(i) for i in all_layers)
    kv_indices = "|".join(str(i) for i in kv_layers)

    part_all = (
        rf"{_LM_LORA_PREFIX}(?=.*\.layers\.({all_indices})\.).*"
        r"\.(?:self_attn\.(?:q_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    )
    if not kv_indices:
        if last_kv is not None and last_kv < total - 1:
            logger.info(
                "KV-aware LoRA: skipping k_proj/v_proj (filter layers %s are KV-shared)",
                all_indices,
            )
        return part_all

    part_kv = (
        rf"{_LM_LORA_PREFIX}(?=.*\.layers\.({kv_indices})\.).*"
        r"\.self_attn\.(?:k_proj|v_proj)$"
    )
    if last_kv is not None and last_kv < total - 1:
        logger.info(
            "KV-aware LoRA: k_proj/v_proj on layers 0–%d; layers %d–%d are KV-shared",
            last_kv,
            last_kv + 1,
            total - 1,
        )
    return f"(?:{part_all}|{part_kv})"


def _extract_layer_filter_from_targets(target_modules: str | list[str]) -> str | None:
    if not isinstance(target_modules, str):
        return None
    m = re.search(r"\.layers\.\(([^)]+)\)", target_modules)
    return m.group(1) if m else None


def adapter_config_needs_kv_patch(target_modules: str | list[str] | None) -> bool:
    if not target_modules:
        return False
    if isinstance(target_modules, list):
        return "k_proj" in target_modules or "v_proj" in target_modules
    if "k_proj" not in target_modules and "v_proj" not in target_modules:
        return False
    # Legacy full-stack regex declares k/v on every layer.
    return bool(re.search(r"self_attn\.\(?:q_proj\|k_proj\|v_proj\|o_proj\)", target_modules))


def patch_peft_config_for_kv_shared(peft_config: Any, model: Any) -> Any:
    """Align adapter_config target_modules with modules that exist in the base model."""
    targets = peft_config.target_modules
    if not adapter_config_needs_kv_patch(targets):
        return peft_config
    layer_filter = _extract_layer_filter_from_targets(targets)
    patched = build_kv_aware_lm_lora_regex(model, layer_filter=layer_filter)
    if patched != targets:
        logger.info("Patched PEFT target_modules for Gemma 4 KV-shared layers")
        peft_config.target_modules = patched
    return peft_config


def rewrite_adapter_config_for_kv_shared(adapter_dir: Path | str, model: Any) -> bool:
    """Fix adapter_config.json on disk after save (or before Hub publish). Returns True if rewritten."""
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return False
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    targets = data.get("target_modules")
    if not adapter_config_needs_kv_patch(targets):
        return False
    layer_filter = _extract_layer_filter_from_targets(targets) if isinstance(targets, str) else None
    data["target_modules"] = build_kv_aware_lm_lora_regex(model, layer_filter=layer_filter)
    cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Rewrote %s for Gemma 4 KV-shared layers", cfg_path)
    return True


def load_gemma4_peft_adapter(base_model: Any, adapter_path: Path | str, **kwargs: Any) -> Any:
    """Load a Gemma 4 LoRA adapter without missing-key warnings on KV-shared layers."""
    from peft import PeftConfig, PeftModel

    adapter_path = str(adapter_path)
    cfg_kw = {k: v for k, v in kwargs.items() if k in ("token", "revision", "cache_dir", "subfolder")}
    peft_config = PeftConfig.from_pretrained(adapter_path, **cfg_kw)
    peft_config = patch_peft_config_for_kv_shared(peft_config, base_model)
    load_kw = {k: v for k, v in kwargs.items() if k not in cfg_kw}
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        config=peft_config,
        **load_kw,
    )


def build_gemma4_lora_config(
    model: Any | None = None,
    *,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: str | list[str] | None = None,
    modules_to_save: list[str] | None = None,
) -> LoraConfig:
    if target_modules is None:
        if model is not None:
            target_modules = build_kv_aware_lm_lora_regex(model)
        else:
            logger.warning(
                "build_gemma4_lora_config without model uses legacy regex; "
                "pass model= for KV-shared-safe targets"
            )
            target_modules = DEFAULT_GEMMA4_LM_LORA_TARGETS
    # Never default-wrap embed_audio: Gemma 4 calls it as embed_audio(inputs_embeds=...)
    # and PEFT AuxiliaryTrainingWrapper.forward(x) raises TypeError.
    if modules_to_save is None:
        modules_to_save = []
    return LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=target_modules,
        modules_to_save=list(modules_to_save) if modules_to_save else None,
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
        return build_kv_aware_lm_lora_regex(model)
    raise RuntimeError("Could not infer language-model LoRA targets from model.named_modules()")


def apply_gemma4_lora(
    model: Any,
    lora_config: LoraConfig,
    *,
    debug_targets: bool = False,
    include_audio_tower: bool = False,
    audio_tower_last_layers: int = 4,
) -> Any:
    targets = lora_config.target_modules
    n = count_lora_target_modules(model, targets)
    if n == 0 and isinstance(targets, str):
        inferred = infer_lm_lora_regex_from_model(model)
        lora_config = build_gemma4_lora_config(
            model=model,
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
        model = get_peft_model(model, lora_config)
    except ValueError as e:
        if "Gemma4ClippableLinear" not in str(e):
            raise
        raise RuntimeError(
            "PEFT failed on Gemma4ClippableLinear. Upgrade peft (>=0.15 recommended) or use "
            "scripts/train_gemma4.py --no-4bit for bf16 LoRA. "
            f"Original error: {e}"
        ) from e
    n_audio = unfreeze_audio_mapper(
        model,
        include_audio_tower=include_audio_tower,
        audio_tower_last_layers=audio_tower_last_layers,
    )
    if n_audio:
        print(
            f"[train] Unfroze {n_audio} audio tensors "
            f"(embed_audio"
            f"{'+ audio_tower' if include_audio_tower else ''}; not PEFT modules_to_save)"
        )
    return model


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

# Gemma 4 projects audio with Gemma4MultimodalEmbedder (`embed_audio`), not
# LLaVA-style `audio_projector` / `multi_modal_projector` (those names are absent).
#
# Do not put embed_audio in PEFT modules_to_save. The parent forward is
#   self.embed_audio(inputs_embeds=audio_outputs.last_hidden_state)
# and PEFT's AuxiliaryTrainingWrapper requires a positional `x` (peft#3191).
PROJECTOR_MODULE_KEYS = ("embed_audio", "audio_projector", "multi_modal_projector")
_AUDIO_LAYER_RE = re.compile(r"audio_tower.*(?:layers|layer|blocks|block)\.(\d+)")


def _audio_tower_layer_index(name: str) -> int | None:
    m = _AUDIO_LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def unfreeze_audio_mapper(
    model: Any,
    *,
    include_audio_tower: bool = False,
    audio_tower_last_layers: int = 4,
) -> int:
    """Mark embed_audio (and optionally last audio_tower layers) trainable after PEFT freeze."""
    max_audio = -1
    if include_audio_tower:
        for name, _ in model.named_parameters():
            idx = _audio_tower_layer_index(name)
            if idx is not None:
                max_audio = max(max_audio, idx)
    first_unfreeze = 0
    if include_audio_tower and audio_tower_last_layers > 0 and max_audio >= 0:
        first_unfreeze = max(0, max_audio - audio_tower_last_layers + 1)
        print(
            f"[train] audio_tower: unfreezing layers {first_unfreeze}–{max_audio} "
            f"(of 0–{max_audio})"
        )
    elif include_audio_tower:
        print("[train] audio_tower: unfreezing all audio_tower parameters")

    n = 0
    for name, param in model.named_parameters():
        if any(k in name for k in PROJECTOR_MODULE_KEYS):
            param.requires_grad_(True)
            n += 1
            continue
        if not include_audio_tower or "audio_tower" not in name or "vision_tower" in name:
            continue
        idx = _audio_tower_layer_index(name)
        if audio_tower_last_layers <= 0 or max_audio < 0:
            param.requires_grad_(True)
            n += 1
        elif idx is not None and idx >= first_unfreeze:
            param.requires_grad_(True)
            n += 1
    return n


def freeze_lm_decoder(
    model: Any,
    *,
    include_audio_tower: bool = False,
    audio_tower_last_layers: int = 4,
) -> Any:
    """Freeze the LM; train embed_audio and optionally last audio_tower layers (asr_safe)."""
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_tensors = unfreeze_audio_mapper(
        model,
        include_audio_tower=include_audio_tower,
        audio_tower_last_layers=audio_tower_last_layers,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0 or trainable_tensors == 0:
        sample = [n for n, _ in list(model.named_parameters())[:12]]
        raise RuntimeError(
            "asr_safe: no parameters matched audio mapper keys "
            f"{PROJECTOR_MODULE_KEYS}. Trainable=0; sample names: {sample}"
        )
    logger.info(
        "[asr_safe] Trainable params: %s / %s (%.2f%%) — audio path",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )
    print(
        f"[asr_safe] Trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.4f}%)"
    )
    return model


def save_projector_checkpoint(model: Any, out_dir: Path | str, *, training_mode: str = "asr_safe") -> None:
    """Save embed_audio weights next to a LoRA adapter or as an asr_safe-only checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainable_audio_tower = any(
        "audio_tower" in n and p.requires_grad for n, p in model.named_parameters()
    )
    state = {}
    for k, v in model.state_dict().items():
        if any(key in k for key in PROJECTOR_MODULE_KEYS):
            state[k] = v.cpu().contiguous()
        elif trainable_audio_tower and "audio_tower" in k:
            state[k] = v.cpu().contiguous()
    if not state:
        raise RuntimeError(
            f"No projector tensors to save (looked for {PROJECTOR_MODULE_KEYS} in state_dict). "
            "Gemma 4 audio mapping lives in embed_audio."
        )
    save_file(state, out_dir / "projector_weights.safetensors")
    (out_dir / "training_mode.json").write_text(
        json.dumps({"training_mode": training_mode, "saved_modules": list(PROJECTOR_MODULE_KEYS)}),
        encoding="utf-8",
    )
    logger.info("Saved projector checkpoint (%d tensors, mode=%s) to %s", len(state), training_mode, out_dir)


def has_projector_weights(adapter_dir: Path | str) -> bool:
    return (Path(adapter_dir) / "projector_weights.safetensors").is_file()


def is_projector_only_checkpoint(adapter_dir: Path | str) -> bool:
    """True when embed_audio was saved without a PEFT adapter_config.json."""
    d = Path(adapter_dir)
    return has_projector_weights(d) and not (d / "adapter_config.json").is_file()


def _match_state_key(key: str, model_keys: set[str]) -> str | None:
    if key in model_keys:
        return key
    prefixes = ("module.", "base_model.model.", "base_model.", "model.model.", "model.")
    variants = [key]
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                variants.append(stripped)
                changed = True
                break
    for v in variants:
        if v in model_keys:
            return v
        for wrapped in (f"model.{v}", f"base_model.model.{v}", f"base_model.{v}"):
            if wrapped in model_keys:
                return wrapped
    return None


def load_projector_checkpoint(model: Any, adapter_dir: Path | str) -> Any:
    """Overlay embed_audio weights onto a base or PeftModel."""
    state = load_file(str(Path(adapter_dir) / "projector_weights.safetensors"))
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for key, tensor in state.items():
        matched = _match_state_key(key, model_keys)
        if matched is not None:
            remapped[matched] = tensor
    if not remapped:
        raise RuntimeError(
            f"Projector checkpoint had {len(state)} tensors but none matched the model. "
            f"Examples: {list(state)[:5]} vs model e.g. "
            f"{[k for k in model_keys if 'embed_audio' in k or 'projector' in k][:8]}"
        )
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    proj_missing = [k for k in missing if any(p in k for p in PROJECTOR_MODULE_KEYS)]
    if unexpected:
        logger.warning("Unexpected keys in projector checkpoint: %s", unexpected[:5])
    if proj_missing:
        logger.warning("Projector keys still at base init after load: %s", proj_missing[:8])
    logger.info(
        "Loaded projector weights (%d tensors applied / %d in file); %d other keys unchanged",
        len(remapped), len(state), len(missing),
    )
    print(f"[eval] Applied {len(remapped)} projector tensors from {adapter_dir}")
    return model


def load_ndizi_checkpoint(base: Any, adapter_dir: Path | str) -> Any:
    """Load asr_safe projector weights and/or a LoRA adapter onto `base`."""
    adapter_dir = Path(adapter_dir)
    if is_projector_only_checkpoint(adapter_dir):
        return load_projector_checkpoint(base, adapter_dir)
    model = load_gemma4_peft_adapter(base, adapter_dir)
    if has_projector_weights(adapter_dir):
        model = load_projector_checkpoint(model, adapter_dir)
    return model


# ── asr_moderate: tail-LoRA on last N decoder layers + full projector save ────

def _count_decoder_layers(model: Any) -> int:
    patterns = (
        re.compile(r"\.language_model\.layers\.(\d+)\."),
        re.compile(r"\.language_model\.model\.layers\.(\d+)\."),
    )
    max_idx = -1
    for name, _ in model.named_modules():
        for pat in patterns:
            m = pat.search(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
                break
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
    """LoRA on the last decoder layers; always includes the last unique-KV layer on Gemma 4 E2B."""
    total = _count_decoder_layers(model)
    last_kv = _last_layer_with_kv_proj(model)
    if last_kv is not None:
        min_tail = total - last_kv  # include layer last_kv (has k_proj/v_proj)
        if num_tail_layers < min_tail:
            print(
                f"[asr_moderate] Raising --tail-lora-layers {num_tail_layers} → {min_tail} "
                f"so LoRA includes unique-KV layer {last_kv} (layers {last_kv + 1}–{total - 1} are KV-shared)"
            )
            num_tail_layers = min_tail
    first_tail = max(0, total - num_tail_layers)
    tail_indices = "|".join(str(i) for i in range(first_tail, total))
    target_regex = build_kv_aware_lm_lora_regex(model, layer_filter=tail_indices)
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
        task_type="CAUSAL_LM",
    )
