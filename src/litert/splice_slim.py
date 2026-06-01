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
) -> Path:
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
        "--prefill_lengths=[128]",
        "--cache_length=2048",
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
    n = path.as_posix().lower()
    if "prefill_decode" in n or ("prefill" in n and "decode" in n):
        return "prefill_decode"
    if "embedder" in n:
        return "embedder"
    if "prefill" in n:
        return "prefill"
    if "decode" in n:
        return "decode"
    return "prefill_decode"


def inventory_from_dump(dump_dir: Path, peek_log: str) -> list[BundleSection]:
    """Build section list from peek log + unpacked files (peek log gives order when parseable)."""
    sections: list[BundleSection] = []

    mentioned: list[Path] = []
    for line in peek_log.splitlines():
        m = re.search(r"(?:data_path|path|file)\s*[:=]\s*['\"]?([^\s'\"]+\.(?:tflite|pb|json|model))", line, re.I)
        if m:
            p = dump_dir / m.group(1)
            if not p.is_file():
                p = next(dump_dir.rglob(Path(m.group(1)).name), None)
            if p and p.is_file():
                mentioned.append(p.resolve())

    seen: set[Path] = set()

    def add_section(sec: BundleSection) -> None:
        key = sec.data_path.resolve()
        if key in seen:
            return
        seen.add(key)
        sections.append(sec)

    for path in mentioned:
        if path.suffix == ".pb":
            add_section(BundleSection("LlmMetadata", path))
        elif path.suffix == ".tflite":
            add_section(
                BundleSection(
                    "TFLiteModel",
                    path,
                    model_type=_infer_tflite_model_type(path),
                )
            )
        elif path.name == "tokenizer.json":
            add_section(BundleSection("HF_Tokenizer", path))

    metadata_pbs = [
        pb
        for pb in dump_dir.rglob("*.pb")
        if "metadata" in pb.name.lower() or "llm" in pb.name.lower()
    ]
    if metadata_pbs:
        llm_pb = max(metadata_pbs, key=lambda p: p.stat().st_size)
        add_section(BundleSection("LlmMetadata", llm_pb))

    for tok in sorted(dump_dir.rglob("tokenizer.json")):
        add_section(BundleSection("HF_Tokenizer", tok))

    for sp in sorted(dump_dir.rglob("*.model")):
        if "tokenizer" in sp.name.lower() or sp.suffix == ".model":
            add_section(BundleSection("SP_Tokenizer", sp))

    for tflite in sorted(dump_dir.rglob("*.tflite")):
        add_section(
            BundleSection(
                "TFLiteModel",
                tflite,
                model_type=_infer_tflite_model_type(tflite),
            )
        )

    if not sections:
        raise RuntimeError(f"No bundle sections inferred from {dump_dir}")
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
        elif st in ("HF_Tokenizer", "HfTokenizer", "HFTokenizer"):
            lines += ["[[section]]", 'section_type = "HF_Tokenizer"', f'data_path = "{rel}"', ""]
        elif st in ("SP_Tokenizer", "SpTokenizer"):
            lines += ["[[section]]", 'section_type = "SP_Tokenizer"', f'data_path = "{rel}"', ""]
        elif st == "TFLiteModel":
            mt = (sec.model_type or "prefill_decode").upper()
            mt_key = "PREFILL_DECODE" if mt == "PREFILL_DECODE" else mt
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
    _run(
        [
            builder,
            "toml",
            "--path",
            str(toml_path),
            "output",
            "--path",
            str(output_litertlm),
        ],
        cwd=toml_path.parent,
    )


def publish_litertlm(local_path: Path, repo_id: str, path_in_repo: str, *, private: bool = False) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Ndizi Swahili ASR LiteRT-LM slim (Google E2B shell + finetuned LLM)",
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
        ft_litertlm = run_finetuned_export(merged_model, export_dir, quantization=quantization)

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


def run_build(args) -> None:
    from src.utils.paths import ARTIFACTS_DIR

    work_dir = Path(getattr(args, "work_dir", None) or ARTIFACTS_DIR / "litert_slim")
    output = build_slim_bundle(
        work_dir,
        merged_model=args.merged_model,
        output_name=args.output_name,
        skip_export=bool(args.skip_export),
        finetuned_litertlm=Path(args.finetuned_litertlm) if args.finetuned_litertlm else None,
        quantization=args.quantization,
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
