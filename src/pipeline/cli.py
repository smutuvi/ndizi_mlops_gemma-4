# src/pipeline/cli.py — argparse entry for Gemma 4 ASR phases.
from __future__ import annotations

import argparse
import sys

from src.eval.normalize import TEXT_NORMALIZE_EVAL_DEFAULT, add_normalize_arg
from src.utils.runtime import apply_model_choice

_PIPELINE_DOC = """
End-to-end fine-tune of Gemma 4 for Swahili ASR on smutuvi/ndizi-1 + smutuvi/ndizi-1-2025.

Adapter-first workflow (QLoRA): publish adapter-only by default; merged weights only
after retention checks pass.

Examples:
  python run_pipeline.py inspect
  python run_pipeline.py prepare --chunk-long-audio
  python scripts/train_gemma4.py --replay-ratio 0.05 --lr 1e-4
  python scripts/publish_gemma4.py --adapter-only
"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_PIPELINE_DOC,
    )

    model_parent = argparse.ArgumentParser(add_help=False)
    model_parent.add_argument(
        "--model",
        default="E2B",
        help="E2B, E4B, or full HF model id (default: E2B).",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="Schema + duration + speaker overlap")

    p_prep = sub.add_parser("prepare", help="Standardise + merge domain datasets")
    p_prep.add_argument("--push", action="store_true", help="Push prepared dataset to Hub.")
    p_prep.add_argument(
        "--retention-datasets",
        nargs="+",
        default=[],
        help="Optional retention Hub datasets (repo or repo:split).",
    )
    p_prep.add_argument("--retention-chunk-test", action="store_true")
    p_prep.add_argument("--chunk-long-audio", action="store_true")
    p_prep.add_argument("--chunk-test", action="store_true")
    p_prep.add_argument(
        "--aggressive-qc",
        action="store_true",
        help="Multi-gate QC on each Hub split before merge (not default).",
    )
    p_prep.add_argument("--qc-use-may6-text-norm", action="store_true")
    p_prep.add_argument("--qc-chunk-long-with-mms-fa", action="store_true")
    p_prep.add_argument("--qc-chunk-seconds", type=float, default=30.0, dest="qc_chunk_seconds")

    p_base = sub.add_parser("baseline", parents=[model_parent], help="Zero-shot WER/CER")
    p_base.add_argument("--with-whisper", action="store_true")
    p_base.add_argument("--batch-size", type=int, default=4)
    p_base.add_argument("--retention-eval", action="store_true")
    add_normalize_arg(p_base)

    p_train = sub.add_parser("train", parents=[model_parent], help="QLoRA fine-tune")
    p_train.add_argument("--retention-datasets", nargs="+", default=[])
    p_train.add_argument("--replay-ratio", type=float, default=0.0)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--epochs", type=float, default=2.0)
    p_train.add_argument("--grad-accum", type=int, default=16)
    p_train.add_argument("--warmup-ratio", type=float, default=0.03)
    p_train.add_argument("--lr-scheduler", type=str, default="cosine")
    p_train.add_argument("--eval-steps", type=int, default=500)
    p_train.add_argument("--eval-max-samples", type=int, default=64)
    p_train.add_argument("--no-train-eval", action="store_true")
    p_train.add_argument("--save-steps", type=int, default=500)
    p_train.add_argument("--save-total-limit", type=int, default=3)
    p_train.add_argument("--no-4bit", action="store_true")
    p_train.add_argument("--lora-target-modules", default=None)
    p_train.add_argument("--debug-lora-targets", action="store_true")
    p_train.add_argument("--peft-clippable-patch", action="store_true")

    p_eval = sub.add_parser("evaluate", parents=[model_parent], help="Finetuned WER/CER or one file")
    p_eval.add_argument("--audio", default=None, help="Transcribe one audio file")
    p_eval.add_argument("--checkpoint", default=None, help="LoRA adapter dir (default: artifacts/checkpoints/best)")
    p_eval.add_argument("--test-datasets", nargs="+", default=None, dest="test_datasets")
    p_eval.add_argument("--output-dir", default=None, dest="output_dir")
    p_eval.add_argument("--reference", default=None, help="Reference text for WER when using --audio")
    p_eval.add_argument("--output", default=None, help="JSON output path for --audio mode")
    p_eval.add_argument("--batch-size", type=int, default=4)
    p_eval.add_argument("--chunk-length-s", type=float, default=None, dest="chunk_length_s")
    p_eval.add_argument("--stride-length-s", type=float, default=None, dest="stride_length_s")
    p_eval.add_argument("--no-auto-chunk", action="store_true")
    p_eval.add_argument("--max-audio-seconds", type=float, default=None, dest="max_audio_seconds")
    p_eval.add_argument("--fp16", action="store_true")
    p_eval.add_argument("--retention-eval", action="store_true")
    add_normalize_arg(p_eval, default=TEXT_NORMALIZE_EVAL_DEFAULT)

    p_pub = sub.add_parser("publish", parents=[model_parent], help="Push to Hub")
    g = p_pub.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter-only", dest="merged", action="store_false")
    g.add_argument("--merged", dest="merged", action="store_true")
    p_pub.add_argument("--max-retention-wer-delta", type=float, default=0.02)
    p_pub.add_argument("--force-merged", action="store_true")

    p_all = sub.add_parser("all", parents=[model_parent], help="Run every phase")
    p_all.add_argument("--with-whisper", action="store_true")
    p_all.add_argument("--push-prepared", action="store_true")
    p_all.add_argument("--chunk-long-audio", action="store_true")
    p_all.add_argument("--chunk-test", action="store_true")
    p_all.add_argument("--aggressive-qc", action="store_true")
    p_all.add_argument("--qc-use-may6-text-norm", action="store_true")
    p_all.add_argument("--qc-chunk-long-with-mms-fa", action="store_true")
    p_all.add_argument("--qc-chunk-seconds", type=float, default=30.0, dest="qc_chunk_seconds")
    p_all.add_argument("--retention-datasets", nargs="+", default=[])
    p_all.add_argument("--retention-eval", action="store_true")
    p_all.add_argument("--replay-ratio", type=float, default=0.0)
    p_all.add_argument("--lr", type=float, default=2e-4)
    p_all.add_argument("--epochs", type=float, default=2.0)
    p_all.add_argument("--grad-accum", type=int, default=16)
    p_all.add_argument("--warmup-ratio", type=float, default=0.03)
    p_all.add_argument("--lr-scheduler", type=str, default="cosine")
    p_all.add_argument("--eval-steps", type=int, default=500)
    p_all.add_argument("--save-steps", type=int, default=500)
    p_all.add_argument("--save-total-limit", type=int, default=3)
    p_all.add_argument("--merged", action="store_true")
    p_all.add_argument("--max-retention-wer-delta", type=float, default=0.02)
    p_all.add_argument("--force-merged", action="store_true")

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "model", None) and args.cmd in (
        "baseline",
        "train",
        "evaluate",
        "publish",
        "all",
    ):
        apply_model_choice(args.model)

    if args.cmd == "inspect":
        from src.data.inspect import run_inspect

        run_inspect()
    elif args.cmd == "prepare":
        from src.data.prepare import run_prepare

        run_prepare(args)
    elif args.cmd == "baseline":
        from src.eval.baseline import run_baseline

        run_baseline(args)
    elif args.cmd == "train":
        from src.training.train import run_train

        run_train(args)
    elif args.cmd == "evaluate":
        from src.eval.finetuned import run_evaluate, run_transcribe_file

        if getattr(args, "audio", None):
            run_transcribe_file(args)
        else:
            run_evaluate(args)
    elif args.cmd == "publish":
        from src.publish.hub import run_publish

        run_publish(args)
    elif args.cmd == "all":
        from src.data.inspect import run_inspect
        from src.data.prepare import run_prepare
        from src.eval.baseline import run_baseline
        from src.eval.finetuned import run_evaluate
        from src.publish.hub import run_publish
        from src.training.train import run_train

        run_inspect()
        prep_args = argparse.Namespace(
            push=args.push_prepared,
            chunk_long_audio=args.chunk_long_audio,
            chunk_test=args.chunk_test,
            retention_datasets=args.retention_datasets,
            retention_chunk_test=False,
            aggressive_qc=getattr(args, "aggressive_qc", False),
            qc_use_may6_text_norm=getattr(args, "qc_use_may6_text_norm", False),
            qc_chunk_long_with_mms_fa=getattr(args, "qc_chunk_long_with_mms_fa", False),
            qc_chunk_seconds=getattr(args, "qc_chunk_seconds", 30.0),
        )
        run_prepare(prep_args)
        base_args = argparse.Namespace(
            with_whisper=args.with_whisper,
            batch_size=4,
            retention_eval=args.retention_eval,
        )
        run_baseline(base_args)
        run_train(args)
        run_evaluate(args)
        run_publish(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
