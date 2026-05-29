# Baseline recipe (~0.40 normalized WER)

This documents the setup that produced **pooled `wer_normalized` ≈ 0.41** on Hub test
(`smutuvi/ndizi-1:test` + `smutuvi/ndizi-1-2025:test`, n=1041, chunk 30s, `jiwer_default`).
See `predictions/metrics.json` (eval dir `gemma4-eval-run-chuncked`) and
`configs/baseline/e2b_wer_0.41_recipe.yaml`.

## Which git version?

| Piece | Commit / tag | Notes |
|-------|----------------|-------|
| **Recorded eval (0.410 norm WER)** | `d1a6559` · tag `baseline-wer-0.41-2026-05-28` | Eval + anti-loop flags added; your saved run did **not** use anti-loop in `run_info` |
| **Before QC prepare** | stop before `194ca2f` | `--aggressive-qc` on prepare changes data and hurts this baseline |
| **Train on server today** | `main` (`4c325cd`+) | Keep **commands** below; use latest code for 4-bit `torch.finfo` / `masked_scatter` fixes |

**Weights are not in git.** `artifacts/checkpoints/best` from that train must be backed up or retrained.

## What likely made WER worse

| Change | Effect |
|--------|--------|
| `prepare_gemma4.py --aggressive-qc` | Drops many train/val rows; different data distribution |
| Retrain overwriting `artifacts/checkpoints/best` | Old adapter lost unless backed up |
| Eval without `--chunk_length_s 30` or `--anti-loop-decode` | Scoring mismatch vs baseline |

QC on **eval only** (`evaluate_gemma4.py --aggressive-qc`) filters test rows before metrics;
it does **not** match the training recipe below.

## Restore data + retrain (no QC)

```bash
cd ~/ndizi_mlops_gemma-4
git pull

# One-shot script:
bash bash_scripts/restore_baseline_0p40_wer.sh
```

Or step by step:

```bash
# 1. Backup current adapter if you might need it
cp -a artifacts/checkpoints/best artifacts/checkpoints/best-qc-run-backup

# 2. Prepare — chunk only, NO --aggressive-qc
python scripts/prepare_gemma4.py --chunk-long-audio --chunk-test

# 3. Train
python scripts/train_gemma4.py \
  --model E2B \
  --replay-ratio 0.05 \
  --lr 1e-4 \
  --epochs 2 \
  --grad-accum 16 \
  --eval-max-samples 64 \
  --save-steps 500

# 4. Eval (match baseline scoring)
python scripts/evaluate_gemma4.py \
  --model E2B \
  --checkpoint artifacts/checkpoints/best \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-baseline-0p40-restore \
  --chunk_length_s 30 \
  --normalize jiwer_default \
  --anti-loop-decode
```

## Recover old weights without retraining

If you pushed the good adapter to the Hub before the QC retrain:

```bash
python scripts/evaluate_gemma4.py \
  --model E2B \
  --checkpoint smutuvi/gemma-4-e2b-sw-asr-ndizi \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-from-hub \
  --chunk_length_s 30 \
  --normalize jiwer_default \
  --anti-loop-decode
```

Or copy a local backup:

```bash
rm -rf artifacts/checkpoints/best
cp -a artifacts/checkpoints/best-before-restore-YYYYMMDD artifacts/checkpoints/best
```

## Revert code only (checkout old scripts)

```bash
cd ~/ndizi_mlops_gemma-4
git fetch origin
git checkout baseline-wer-0.41-2026-05-28   # scripts at d1a6559 era
# For training on 4-bit, merge fixes: git checkout main -- src/models/gemma4_lora.py src/training/train.py
```

## Git reference

- Tag: `baseline-wer-0.41-2026-05-28` → `d1a6559`
- QC on **prepare**: `194ca2f` — avoid for this baseline
- Train crash fixes: `2ee05fc` … `4c325cd` on `main`
