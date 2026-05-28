## ndizi_mlops_gemma-4

Adapter-first fine-tuning + evaluation pipeline for **Gemma 4 E2B** Swahili ASR on:

- `smutuvi/ndizi-1`
- `smutuvi/ndizi-1-2025`

### Quickstart

Install:

```bash
python -m pip install -r requirements.txt
```

Prepare merged dataset artifacts (**no** `--aggressive-qc` for the ~0.40 WER baseline):

```bash
python run_pipeline.py prepare --chunk-long-audio --chunk-test
```

Train LoRA adapter (baseline recipe; see `docs/BASELINE_0.40_WER.md`):

```bash
python scripts/train_gemma4.py \
  --model E2B \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  --epochs 2 \
  --grad-accum 16 \
  --eval-max-samples 64 \
  --save-steps 500
```

To restore after a QC retrain hurt WER: `bash bash_scripts/restore_baseline_0p40_wer.sh`

Evaluate finetuned adapter on Hub test splits (writes `metrics.json`, `predictions.json`, `predictions.csv`):

```bash
python scripts/evaluate_gemma4.py \
  --checkpoint artifacts/checkpoints/best \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-e2b-finetuned \
  --chunk_length_s 30 \
  --normalize jiwer_default \
  --anti-loop-decode
```

Baseline (pre-finetune) evaluation:

```bash
python scripts/baseline_gemma4.py \
  --model E2B \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-e2b-baseline \
  --chunk_length_s 30 \
  --normalize jiwer_default
```

### Notes

- Training checkpoints and datasets are stored under `artifacts/` and are **not** meant to be committed.
- For long audio, evaluation supports fixed-window chunking (`--chunk_length_s 30`).

