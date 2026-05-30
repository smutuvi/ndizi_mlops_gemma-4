# Baseline ~0.41 normalized WER (E2B QLoRA)

Recorded result: **pooled `wer_normalized` = 0.4098** on Hub test  
(`smutuvi/ndizi-1:test` + `smutuvi/ndizi-1-2025:test`, n=1041, `--chunk_length_s 30`, `--normalize jiwer_default`).

Source file: `predictions/metrics.json` (copy: `configs/baseline/metrics_reference.json`).

## What is *not* in git

| Artifact | Path |
|----------|------|
| Trained LoRA adapter | `artifacts/checkpoints/best` |
| Prepared training data | `artifacts/prepared_dataset` |

Re-running train **overwrites** the adapter unless you back it up. To **re-score** the same ~0.41 number, you only need the saved checkpoint + the exact eval command below.

## Exact eval (re-score existing checkpoint)

Matches `metrics.json` / `run_info` (no `--anti-loop-decode`):

```bash
cd ~/ndizi_mlops_gemma-4
git pull   # needs jiwer normalize fix on main

bash bash_scripts/eval_baseline_0p40_exact.sh
```

Or manually:

```bash
python scripts/evaluate_gemma4.py \
  --model E2B \
  --checkpoint artifacts/checkpoints/best \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-eval-run-chuncked \
  --chunk_length_s 30 \
  --batch-size 4 \
  --normalize jiwer_default
```

Compare `eval/gemma4-eval-run-chuncked/metrics.json` → `pooled.wer_normalized` to **0.4098**.

## Full pipeline (retrain to approximate ~0.41)

**Do not** use `--aggressive-qc` on prepare.

```bash
bash bash_scripts/restore_baseline_0p40_wer.sh
```

Steps inside that script:

1. **Prepare** — MMS-FA chunk train/val + test, no QC:

   ```bash
   python scripts/prepare_gemma4.py --chunk-long-audio --chunk-test
   ```

2. **Train** — 4-bit QLoRA (default), replay 5%, lr 1e-4, 2 epochs:

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

   Requires **main @ `323f948` or later** (4-bit audio-tower `torch.finfo` patches).

3. **Eval** — use `eval_baseline_0p40_exact.sh` for the recorded recipe, or add `--anti-loop-decode` to reduce loops (may shift WER slightly).

## Git code reference

| Piece | Commit / tag |
|-------|----------------|
| Eval + anti-loop flags era | `baseline-wer-0.41-2026-05-28` → `d1a6559` |
| Train (4-bit fixes) | `main` → `323f948+` |
| Avoid QC on prepare | before `194ca2f` |
| jiwer 3.x scoring | `c57703b` (`src/eval/normalize.py`) |

Monolith before refactor: `run_pipeline.py.bak` (not the commit that produced this metric; modular pipeline above did).

## Recover weights without retraining

```bash
# If you backed up before QC retrain:
cp -a artifacts/checkpoints/best-before-restore-YYYYMMDD artifacts/checkpoints/best

# Or Hub adapter (if published):
python scripts/evaluate_gemma4.py \
  --model E2B \
  --checkpoint smutuvi/gemma-4-e2b-sw-asr-ndizi \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/gemma4-from-hub \
  --chunk_length_s 30 \
  --batch-size 4 \
  --normalize jiwer_default
```

## Why later runs scored worse

- `prepare --aggressive-qc` changed training data.
- Retrain overwrote `artifacts/checkpoints/best`.
- Eval without `--chunk_length_s 30` on long test clips.
- Mid-training `eval_wer` (~0.70) uses argmax on 64 val rows — **not** comparable to this Hub `generate()` + jiwer eval.
