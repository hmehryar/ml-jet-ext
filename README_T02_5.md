# T02.5 — Full-epoch GPU and legacy validation

Based on the verified T02.4 code interface. This patch adds **tests only**, preserving all existing training and model implementation files.

## 1. Legacy regression (login node or GPU node)

```bash
python -m pytest tests/test_t02_5_legacy_regression.py -v
python -m pytest tests/test_moment_loader.py tests/test_t02_mode_dispatch.py tests/test_t02_vit_moment_heads.py tests/test_t02_moment_training.py tests/test_t02_5_legacy_regression.py -v
```

The legacy regression uses an 8-sample synthetic three-head data source. It is **not** a historical-checkpoint or complete legacy-dataset replay.

## 2. Full epoch (GPU allocation required)

```bash
export DATASET_ROOT=/rs/rs_grp_mljet/ml-jet-ext/pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5_normalized
PYTHONPATH="$PWD" python tests/validate_t02_5_full_epoch.py \
    --config config/config_moment_example.yaml \
    --workers 0 \
    --batch-size 32 \
    --moments 1 2 \
    --output training_output/t02_5_m1_m2
```

This executes **all 350 training batches and 44 validation batches** for 11,200 train / 1,400 val samples (batch size 32; 44 validation batches because final batch has 24). No held-out test evaluation occurs.

Outputs: `moment_epoch_metrics.csv`, `last_model.pth`, `best_model.pth` in the output directory. The script restores model and optimizer, compares parameter values and predictions bit-for-bit on the same evaluation batch, then checks another optimizer step. The saved checkpoint is unchanged by the resume test.

Use a fresh output directory each run. Be sure the chosen Slurm allocation provides sufficient GPU memory, CPU cores and walltime. For worker restrictions use `--workers 0`.

## 3. Remaining before T02 closure

- Run the two legacy regression tests on HPC and confirm the historical legacy dataset still loads if available.
- Run the full epoch on GPU and inspect per-head metrics / confusion matrices.
- Check resume across a *subsequent epoch* and scheduler consistency separately: existing `run_moment_training_loop` accepts `start_epoch`, but does not internally reconstruct checkpoint state; this validation only proves state restoration and a resumed optimizer step.
- Run M1 and M1–M5 end-to-end after M1–M2, then decide whether to start final ablations.
