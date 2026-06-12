# src/litert/splice_slim.py — build ~2.6GB LiteRT-LM bundle: Google E2B base + Ndizi LLM weights.
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Official on-device E2B shell (audio/embedder/tokenizer/metadata).
BASE_LITERT_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
BASE_LITERT_FILE = "gemma-4-E2B-it.litertlm"

DEFAULT_BASE_MODEL = "google/gemma-4-E2B-it"
DEFAULT_ADAPTER = "smutuvi/gemma-4-e2b-sw-asr-ndizi"
DEFAULT_MERGED_MODEL = "smutuvi/gemma-4-e2b-sw-asr-ndizi-merged"
DEFAULT_HUB_REPO = "smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm-slim"
DEFAULT_OUTPUT_NAME = "gemma-4-e2b-sw-asr-ndizi-slim.litertlm"


@dataclass
class BundleSection:
    section_type: str
    data_path: Path
    model_type: str | None = None
    additional_metadata: list[tuple[str, str]] = field(default_factory=list)


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    print("[cmd]", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    return proc.stdout or ""


def _which(name: str) -> str:
    from shutil import which

    path = which(name)
    if not path:
        raise RuntimeError(
            f"'{name}' not found on PATH. Install with:\n"
            "  pip install litert-lm-builder litert-torch-nightly\n"
            "  # peek/builder CLIs: litert-lm-peek, litert-lm-builder"
        )
    return path


def merge_lora_adapter(
    base_model: str,
    adapter: str,
    output_dir: Path,
    *,
    token: str | None = None,
) -> Path:
    """Merge a LoRA/PEFT adapter into the base model and save the result.

    Skips if output_dir already contains model weights (delete to re-run).

    Args:
        base_model: HF repo ID or local path for the base model.
        adapter:    HF repo ID or local path for the LoRA adapter.
        output_dir: Where to save the merged model.
        token:      HuggingFace token (needed for gated models like Gemma).

    Returns:
        output_dir (Path)
    """
    import shutil as _shutil

    output_dir = Path(output_dir)
    weight_files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("pytorch_model*.bin"))
    if weight_files:
        print(f"[merge] {output_dir} already has weights — skipping merge (delete to redo).")
        return output_dir

    print(f"[merge] Loading base model: {base_model}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )

    print(f"[merge] Applying LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter, token=token)

    print("[merge] Merging adapter weights into base model …")
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] Saving merged model → {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    print("[merge] Saving tokenizer …")
    AutoTokenizer.from_pretrained(base_model, token=token).save_pretrained(output_dir)

    # Copy processor / preprocessor configs (needed for audio/vision pipelines).
    print("[merge] Copying processor configs from base model …")
    try:
        from huggingface_hub import hf_hub_download

        for fname in [
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ]:
            try:
                src = hf_hub_download(base_model, fname, token=token)
                _shutil.copy(src, output_dir / fname)
            except Exception:
                pass
    except Exception:
        pass

    print(f"[merge] ✓ Merge complete → {output_dir}")
    return output_dir


def download_base_litertlm(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=BASE_LITERT_REPO,
        filename=BASE_LITERT_FILE,
        local_dir=str(dest_dir),
    )
    return Path(path)


def peek_unpack(litertlm: Path, dump_dir: Path) -> str:
    dump_dir.mkdir(parents=True, exist_ok=True)
    if any(dump_dir.iterdir()):
        shutil.rmtree(dump_dir)
        dump_dir.mkdir(parents=True)
    peek = _which("litert-lm-peek")
    log = _run([peek, "--litertlm_file", str(litertlm), "--dump_files_dir", str(dump_dir)])
    (dump_dir / "peek_stdout.txt").write_text(log, encoding="utf-8")
    return log


def find_prefill_decode_tflite(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = list(root.rglob("*.tflite"))
    if not candidates:
        raise FileNotFoundError(f"No .tflite under {root}")

    def score(p: Path) -> int:
        n = p.as_posix().lower()
        s = 0
        if "prefill_decode" in n or "prefill-decode" in n:
            s += 10
        if "prefill" in n and "decode" in n:
            s += 8
        if "tf_lite_prefill_decode" in n:
            s += 10
        if "prefill" in n:
            s += 3
        if "decode" in n and "prefill" not in n:
            s += 1
        if "embedder" in n or "vision" in n or "audio" in n:
            s -= 5
        return s

    ranked = sorted(candidates, key=lambda p: (-score(p), len(p.name)))
    best = ranked[0]
    if score(best) < 3:
        raise FileNotFoundError(
            f"Could not identify prefill_decode .tflite under {root}. "
            f"Candidates: {[c.relative_to(root) for c in candidates[:8]]}"
        )
    return best


def run_finetuned_export(
    merged_model: str,
    export_dir: Path,
    *,
    quantization: str = "dynamic_wi4_afp32",
    cache_length: int = 1024,
    prefill_lengths: str = "[64]",
) -> Path:
    """Export the merged model to a .litertlm bundle.

    cache_length and prefill_lengths are kept small (1024 / [64]) to ensure
    the prefill_decode TFLite stays under the 2 GB FlatBuffers limit, which
    litert-lm-builder requires.  The community model uses the same settings.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    if any(export_dir.iterdir()):
        shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        "-m",
        "litert_torch.generative.export_hf",
        merged_model,
        str(export_dir),
        "--task=image_text_to_text",
        f"--quantization_recipe={quantization}",
        "--externalize_embedder=True",
        "--bundle_litert_lm=True",
        "--use_jinja_template=True",
        "--litert_lm_model_type_override=gemma4",
        "--export_vision_encoder=False",
        "--keep_temporary_files=True",
        f"--prefill_lengths={prefill_lengths}",
        f"--cache_length={cache_length}",
    ]
    _run(cmd)

    bundles = list(export_dir.rglob("*.litertlm"))
    if not bundles:
        raise FileNotFoundError(f"export_hf finished but no .litertlm under {export_dir}")
    if len(bundles) > 1:
        bundles.sort(key=lambda p: p.stat().st_size, reverse=True)
        print(f"[warn] multiple .litertlm found; using largest: {bundles[0]}")
    return bundles[0]


def _rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _infer_tflite_model_type(path: Path) -> str:
    """Return the builder TOML model_type for a TFLite file.

    litert-lm-peek names dump files as:
        Section{N}_TFLiteModel_tf_lite_{type}.tflite

    get_enum_from_tf_free_value() lowercases the TOML value and prepends
    "tf_lite_", so the TOML model_type must be {type} in UPPERCASE.
    We extract {type} directly from the filename for accuracy.
    """
    # Primary: extract from peek-generated filename (most reliable)
    m = re.search(r"tf_lite_(\w+)", path.stem, re.I)
    if m:
        return m.group(1).upper()

    # Fallback: keyword matching for files not following peek naming
    n = path.as_posix().lower()
    if "per_layer_embedder" in n:
        return "PER_LAYER_EMBEDDER"
    if "prefill_decode" in n or ("prefill" in n and "decode" in n):
        return "PREFILL_DECODE"
    if "mtp_drafter" in n:
        return "MTP_DRAFTER"
    if "audio_encoder_hw" in n:
        return "AUDIO_ENCODER_HW"
    if "audio_adapter" in n:
        return "AUDIO_ADAPTER"
    if "end_of_audio" in n:
        return "END_OF_AUDIO"
    if "vision_encoder" in n:
        return "VISION_ENCODER"
    if "vision_adapter" in n:
        return "VISION_ADAPTER"
    if "end_of_vision" in n:
        return "END_OF_VISION"
    if "embedder" in n:
        return "EMBEDDER"
    return "PREFILL_DECODE"


def inventory_from_dump(dump_dir: Path, peek_log: str) -> list[BundleSection]:
    """Build section list from peek log + unpacked files.

    The peek log is parsed line-by-line; when a model_type appears near a
    data_path the two are associated so we don't have to guess from filenames.
    Filename-based inference is the fallback.
    """
    sections: list[BundleSection] = []
    seen: set[Path] = set()

    def add_section(sec: BundleSection) -> None:
        key = sec.data_path.resolve()
        if key in seen:
            return
        seen.add(key)
        sections.append(sec)

    def resolve_path(raw: str) -> Path | None:
        p = dump_dir / raw
        if p.is_file():
            return p.resolve()
        hit = next(dump_dir.rglob(Path(raw).name), None)
        return hit.resolve() if hit else None

    # ── Pass 1: parse peek log, capturing model_type from context ────────────
    # The peek log groups info about each section across a few lines.
    # We scan with a small sliding context window so that a model_type line
    # appearing before/after the data_path line is still associated correctly.
    _MODEL_TYPE_RE = re.compile(r"model[_\s]type\s*[:=]\s*['\"]?(\w+)['\"]?", re.I)
    _DATA_PATH_RE  = re.compile(
        r"(?:data_path|path|file)\s*[:=]\s*['\"]?([^\s'\"]+\.(?:tflite|pb|json|model))", re.I
    )

    lines = peek_log.splitlines()
    # Build list of (line_index, model_type) and (line_index, data_path) matches,
    # then pair them by proximity.
    mt_hits: list[tuple[int, str]] = []
    dp_hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        mt = _MODEL_TYPE_RE.search(line)
        if mt:
            mt_hits.append((i, mt.group(1)))
        dp = _DATA_PATH_RE.search(line)
        if dp:
            dp_hits.append((i, dp.group(1)))

    # For each data_path hit find the nearest preceding model_type (within 5 lines).
    used_mt: set[int] = set()
    for dp_idx, raw_path in dp_hits:
        p = resolve_path(raw_path)
        if p is None:
            continue
        # Find closest model_type before this line.
        best_mt: str | None = None
        best_dist = 6
        for mt_idx, mt_val in mt_hits:
            dist = dp_idx - mt_idx
            if 0 <= dist < best_dist and mt_idx not in used_mt:
                best_dist = dist
                best_mt = mt_val
        if best_mt and best_dist < 6:
            used_mt.add(mt_hits[[i for i, (mi, _) in enumerate(mt_hits) if mi == dp_idx - best_dist][0]][0])

        if p.suffix == ".pb":
            add_section(BundleSection("LlmMetadata", p))
        elif p.suffix == ".tflite":
            mt = best_mt or _infer_tflite_model_type(p)
            add_section(BundleSection("TFLiteModel", p, model_type=mt))
        elif p.name == "tokenizer.json":
            add_section(BundleSection("HF_Tokenizer", p))

    # ── Pass 2: scan dump dir for anything the log missed ────────────────────
    # LlmMetadata: peek dumps as LlmMetadataProto.pbtext (binary proto, .pbtext ext)
    # or as legacy *.pb.
    metadata_candidates = sorted(
        list(dump_dir.rglob("*.pbtext")) + list(dump_dir.rglob("*.pb")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    metadata_candidates = [
        p for p in metadata_candidates
        if "metadata" in p.name.lower() or "llm" in p.name.lower()
    ]
    if metadata_candidates:
        add_section(BundleSection("LlmMetadata", metadata_candidates[0]))

    # HF_Tokenizer: peek dumps as Section{N}_HF_Tokenizer_Zlib.zlib
    # The builder only accepts *uncompressed* tokenizer JSON as section_type "HF_Tokenizer"
    # (it compresses internally).  Decompress the .zlib so the builder can read it.
    for zlib_path in sorted(dump_dir.rglob("*HF_Tokenizer*.zlib")):
        try:
            import zlib as _zlib
            raw = _zlib.decompress(zlib_path.read_bytes())
            tok_json = zlib_path.parent / "tokenizer_hf.json"
            tok_json.write_bytes(raw)
            print(f"[info] Decompressed {zlib_path.name} → {tok_json.name} ({len(raw)} bytes)")
            add_section(BundleSection("HF_Tokenizer", tok_json))
        except Exception as _e:
            print(f"[warn] Could not decompress {zlib_path}: {_e} — skipping tokenizer")

    # Plain JSON tokenizer (older bundles or export output)
    for tok in sorted(dump_dir.rglob("tokenizer.json")):
        add_section(BundleSection("HF_Tokenizer", tok))

    for sp in sorted(dump_dir.rglob("*.model")):
        if "tokenizer" in sp.name.lower() or sp.suffix == ".model":
            add_section(BundleSection("SP_Tokenizer", sp))

    for tflite in sorted(dump_dir.rglob("*.tflite")):
        add_section(BundleSection("TFLiteModel", tflite, model_type=_infer_tflite_model_type(tflite)))

    if not sections:
        raise RuntimeError(f"No bundle sections inferred from {dump_dir}")

    # ── Reorder: LlmMetadata → Tokenizer → TFLiteModel (builder may be order-sensitive) ──
    _ORDER = {"LlmMetadata": 0, "HF_Tokenizer_Zlib": 1, "HF_Tokenizer": 1,
              "SP_Tokenizer": 1, "TFLiteModel": 2}
    sections.sort(key=lambda s: _ORDER.get(s.section_type, 99))

    # ── Sanity checks ────────────────────────────────────────────────────────
    section_types = {s.section_type for s in sections}
    tflite_types  = {s.model_type for s in sections if s.section_type == "TFLiteModel"}

    if "LlmMetadata" not in section_types:
        print(
            "[WARN] No LlmMetadata section found — model will crash at load time!\n"
            "       Peek should dump LlmMetadataProto.pbtext; check your dump dir."
        )
    else:
        print("[info] LlmMetadata section present ✓")

    if not section_types & {"HF_Tokenizer", "HF_Tokenizer_Zlib", "SP_Tokenizer"}:
        print(
            "[WARN] No tokenizer section found — model will crash at load time!\n"
            "       Peek should dump *HF_Tokenizer*.zlib; check your dump dir."
        )
    else:
        print("[info] Tokenizer section present ✓")

    if "AUDIO_ENCODER_HW" not in tflite_types:
        print(
            "[warn] No AUDIO_ENCODER_HW section found — ASR will fail at runtime.\n"
            "       Check that the base community bundle contains an audio encoder TFLite."
        )
    else:
        print("[info] Audio encoder section present ✓")

    return sections


def write_bundle_toml(sections: list[BundleSection], toml_path: Path) -> None:
    toml_dir = toml_path.parent
    lines = [
        '[system_metadata]',
        "entries = [",
        '  { key = "author", value_type = "String", value = "smutuvi/ndizi" },',
        '  { key = "base_litert", value_type = "String", value = "litert-community/gemma-4-E2B-it-litert-lm" },',
        '  { key = "finetune", value_type = "String", value = "smutuvi/gemma-4-e2b-sw-asr-ndizi-merged" },',
        '  { key = "bundle_kind", value_type = "String", value = "spliced_llm_prefill_decode" },',
        "]",
        "",
    ]
    for sec in sections:
        rel = _rel(sec.data_path, toml_dir)
        st = sec.section_type
        if st == "LlmMetadata":
            lines += ["[[section]]", 'section_type = "LlmMetadata"', f'data_path = "{rel}"', ""]
        elif st in ("HF_Tokenizer", "HfTokenizer", "HFTokenizer", "HF_Tokenizer_Zlib"):
            lines += ["[[section]]", 'section_type = "HF_Tokenizer"', f'data_path = "{rel}"', ""]
        elif st in ("SP_Tokenizer", "SpTokenizer"):
            lines += ["[[section]]", 'section_type = "SP_Tokenizer"', f'data_path = "{rel}"', ""]
        elif st == "TFLiteModel":
            # model_type is already in the correct TOML format from _infer_tflite_model_type().
            # get_enum_from_tf_free_value() lowercases and prepends "tf_lite_", so we pass
            # the UPPERCASE suffix (e.g. "AUDIO_ENCODER_HW" → "tf_lite_audio_encoder_hw").
            mt_key = (sec.model_type or "PREFILL_DECODE").upper()
            lines += [
                "[[section]]",
                'section_type = "TFLiteModel"',
                f'model_type = "{mt_key}"',
                f'data_path = "{rel}"',
            ]
            for k, v in sec.additional_metadata:
                lines.append("additional_metadata = [")
                lines.append(f'  {{ key = "{k}", value_type = "String", value = "{v}" }},')
                lines.append("]")
            lines.append("")
        else:
            print(f"[warn] skipping unknown section type {st!r} ({rel})")

    toml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {toml_path}")


def build_litertlm_from_toml(toml_path: Path, output_litertlm: Path) -> None:
    builder = _which("litert-lm-builder")
    if output_litertlm.exists():
        output_litertlm.unlink()

    # Print TOML so any misconfiguration is visible in the log.
    print(f"\n[toml] {toml_path}\n{toml_path.read_text()}")

    cmd = [
        builder,
        "toml",
        "--path",
        str(toml_path),
        "output",
        "--path",
        str(output_litertlm),
    ]
    print("[cmd]", " ".join(cmd), flush=True)

    # Stream builder output directly — do NOT capture — so errors are visible.
    import subprocess as _sp
    result = _sp.run(cmd, cwd=str(toml_path.parent))
    if result.returncode != 0:
        raise RuntimeError(
            f"litert-lm-builder failed (exit {result.returncode}). "
            f"Check the TOML printed above for wrong model_type or missing files."
        )


def publish_litertlm(
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    *,
    private: bool = False,
    commit_message: str = "Ndizi Swahili ASR LiteRT-LM slim (Google E2B shell + finetuned LLM)",
) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Published https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}")


def write_readme(work_dir: Path, repo_id: str, filename: str) -> None:
    readme = work_dir / "README.md"
    readme.write_text(
        f"""---
base_model: google/gemma-4-E2B-it
license: gemma
tags:
  - litert-lm
  - automatic-speech-recognition
  - swahili
  - gemma-4
datasets:
  - smutuvi/ndizi-1
  - smutuvi/ndizi-1-2025
---

# Gemma 4 E2B Ndizi Swahili ASR (LiteRT-LM, slim)

On-device bundle (~2.6 GB target): **LiteRT shell** from [{BASE_LITERT_REPO}](https://huggingface.co/{BASE_LITERT_REPO})
with **prefill/decode LLM weights** from [{DEFAULT_MERGED_MODEL}](https://huggingface.co/{DEFAULT_MERGED_MODEL}).

- File: `{filename}`
- Adapter (GPU/Colab): [smutuvi/gemma-4-e2b-sw-asr-ndizi](https://huggingface.co/smutuvi/gemma-4-e2b-sw-asr-ndizi)
- Full export (~5 GB): [smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm](https://huggingface.co/smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm)

## Inference (Swahili ASR)

Use the same audio-first chat turn and Swahili ASR instruction as training (`ndizi_mlops_gemma-4`).

## Build

Reproduced with `python scripts/build_litert_lm_slim.py` in `ndizi_mlops_gemma-4`.
""",
        encoding="utf-8",
    )


def build_slim_bundle(
    work_dir: Path,
    *,
    merged_model: str = DEFAULT_MERGED_MODEL,
    output_name: str = DEFAULT_OUTPUT_NAME,
    skip_export: bool = False,
    finetuned_litertlm: Path | None = None,
    quantization: str = "dynamic_wi4_afp32",
    cache_length: int = 1024,
    prefill_lengths: str = "[64]",
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    base_dir = work_dir / "base"
    base_litertlm = download_base_litertlm(base_dir)
    print(f"Base bundle: {base_litertlm} ({base_litertlm.stat().st_size / 1e9:.2f} GB)")

    base_dump = work_dir / "base_unpack"
    base_log = peek_unpack(base_litertlm, base_dump)

    if skip_export:
        if finetuned_litertlm is None:
            candidates = list(work_dir.rglob("*.litertlm"))
            candidates = [c for c in candidates if c.resolve() != base_litertlm.resolve()]
            if not candidates:
                raise ValueError("--skip-export requires --finetuned-litertlm or an existing export .litertlm")
            finetuned_litertlm = max(candidates, key=lambda p: p.stat().st_size)
        ft_litertlm = Path(finetuned_litertlm)
    else:
        export_dir = work_dir / "finetuned_export"
        ft_litertlm = run_finetuned_export(
            merged_model, export_dir,
            quantization=quantization,
            cache_length=cache_length,
            prefill_lengths=prefill_lengths,
        )

    print(f"Finetuned export bundle: {ft_litertlm} ({ft_litertlm.stat().st_size / 1e9:.2f} GB)")

    ft_dump = work_dir / "finetuned_unpack"
    peek_unpack(ft_litertlm, ft_dump)

    base_prefill = find_prefill_decode_tflite(base_dump)
    ft_prefill = find_prefill_decode_tflite(ft_dump)
    print(f"Replace prefill_decode:\n  base: {base_prefill}\n  with: {ft_prefill}")
    shutil.copy2(ft_prefill, base_prefill)

    bundle_dir = work_dir / "bundle_staging"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(base_dump, bundle_dir)

    sections = inventory_from_dump(bundle_dir, base_log)
    manifest = work_dir / "bundle_manifest.json"
    manifest.write_text(
        json.dumps([{**asdict(s), "data_path": str(s.data_path)} for s in sections], indent=2),
        encoding="utf-8",
    )

    toml_path = bundle_dir / "bundle.toml"
    write_bundle_toml(sections, toml_path)

    output_litertlm = work_dir / output_name
    build_litertlm_from_toml(toml_path, output_litertlm)
    size_gb = output_litertlm.stat().st_size / 1e9
    print(f"Output: {output_litertlm} ({size_gb:.2f} GB)")
    if size_gb > 3.2:
        print(
            "[warn] Bundle is still larger than ~2.6 GB. "
            "It may include extra graphs; test on device and consider wi4 export or manual peek review."
        )
    return output_litertlm


def build_official_shell_bundle(work_dir: Path, *, output_name: str = DEFAULT_OUTPUT_NAME) -> Path:
    """Copy Google's ~2.6 GB E2B .litertlm (loads on typical phones; not Ndizi-finetuned)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    base_litertlm = download_base_litertlm(work_dir / "base")
    output_litertlm = work_dir / output_name
    if output_litertlm.exists():
        output_litertlm.unlink()
    shutil.copy2(base_litertlm, output_litertlm)
    size_gb = output_litertlm.stat().st_size / 1e9
    print(f"Official shell (low RAM): {output_litertlm} ({size_gb:.2f} GB)")
    print(
        "[info] This is the stock google/gemma-4-E2B-it LiteRT bundle, not Ndizi-finetuned. "
        "Use GPU/server ASR with smutuvi/gemma-4-e2b-sw-asr-ndizi-merged for Ndizi WER."
    )
    return output_litertlm


def write_readme_official_shell(work_dir: Path, repo_id: str, filename: str) -> None:
    readme = work_dir / "README.md"
    readme.write_text(
        f"""---
base_model: google/gemma-4-E2B-it
license: gemma
tags:
  - litert-lm
  - automatic-speech-recognition
  - swahili
  - gemma-4
  - low-ram
---

# Gemma 4 E2B LiteRT-LM (low RAM, official shell)

**~2.6 GB** — same file as [{BASE_LITERT_REPO}](https://huggingface.co/{BASE_LITERT_REPO}) (`{BASE_LITERT_FILE}`),
renamed for Sikia as `{filename}`.

Use this on phones that **hang or OOM** on the ~4–5 GB custom finetuned LiteRT builds.

## Ndizi Swahili ASR quality

This bundle is **not** the Ndizi fine-tune. For production ASR WER, use:

- **GPU / server:** [smutuvi/gemma-4-e2b-sw-asr-ndizi-merged](https://huggingface.co/smutuvi/gemma-4-e2b-sw-asr-ndizi-merged) or LoRA [smutuvi/gemma-4-e2b-sw-asr-ndizi](https://huggingface.co/smutuvi/gemma-4-e2b-sw-asr-ndizi)
- **Heavy on-device finetuned LiteRT:** [smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm](https://huggingface.co/smutuvi/gemma-4-e2b-sw-asr-ndizi-litert-lm) (~5 GB) or spliced rebuild (~4 GB)

A finetuned bundle near **2.6 GB** needs in-container splice (not unpack/repack); track `scripts/build_litert_lm_slim.py` in `ndizi_mlops_gemma-4`.
""",
        encoding="utf-8",
    )


def run_build(args) -> None:
    from src.utils.paths import ARTIFACTS_DIR

    work_dir = Path(getattr(args, "work_dir", None) or ARTIFACTS_DIR / "litert_slim")

    # ── Optional merge step ───────────────────────────────────────────────────
    if getattr(args, "merge", False):
        token = getattr(args, "hf_token", None)
        merged_dir = work_dir / "merged_model"
        merge_lora_adapter(
            base_model=getattr(args, "base_model", DEFAULT_BASE_MODEL),
            adapter=getattr(args, "adapter", DEFAULT_ADAPTER),
            output_dir=merged_dir,
            token=token,
        )
        # Override merged_model to point at our freshly merged local copy.
        args.merged_model = str(merged_dir)

    if getattr(args, "official_shell", False):
        output = build_official_shell_bundle(work_dir, output_name=args.output_name)
        if args.upload:
            publish_litertlm(
                output,
                args.hub_repo,
                args.output_name,
                commit_message="Low-RAM official E2B LiteRT shell (~2.6 GB)",
            )
            write_readme_official_shell(work_dir, args.hub_repo, args.output_name)
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(work_dir / "README.md"),
                path_in_repo="README.md",
                repo_id=args.hub_repo,
                repo_type="model",
                commit_message="Document low-RAM official shell vs finetuned bundles",
            )
        return

    output = build_slim_bundle(
        work_dir,
        merged_model=args.merged_model,
        output_name=args.output_name,
        skip_export=bool(args.skip_export),
        finetuned_litertlm=Path(args.finetuned_litertlm) if args.finetuned_litertlm else None,
        quantization=args.quantization,
        cache_length=getattr(args, "cache_length", 1024),
        prefill_lengths=getattr(args, "prefill_lengths", "[64]"),
    )

    if args.upload:
        publish_litertlm(output, args.hub_repo, args.output_name)
        write_readme(work_dir, args.hub_repo, args.output_name)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(work_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=args.hub_repo,
            repo_type="model",
            commit_message="Add model card for slim LiteRT-LM bundle",
        )
