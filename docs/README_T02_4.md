# T02.4: Moment training & evaluation integration

Patch built against the previously supplied repository ZIP with T02.1–T02.3 patches applied. **Review the diff before overwriting your HPC checkout.**

Files:
- `train_utils/moment_training.py`: new two-target train/eval/loop, macro metrics, CSV logs, best/last checkpoints.
- `train_utils/train_epoch.py`: optional `dataset_type='moment'` dispatch, defaults to original legacy implementation.
- `train_utils/evaluate.py`: optional `dataset_type='moment'` dispatch, otherwise original legacy implementation.
- `train_utils/training_loop.py`: new model factory routing by dataset type; moment-specific epoch loop while preserving original legacy loggers and early stopping.
- `tests/test_t02_moment_training.py`: three synthetic tests.
- `tests/smoke_t02_4_real_batch.py`: actual dataset, one batch through optimizer step and validation.

## Installation

```bash
# From repository root, inspect ZIP before extracting
unzip -l ML_JET_Ext_T02_4_Training_Integration.zip
# Save a copy of modified files, then extract
unzip -o ML_JET_Ext_T02_4_Training_Integration.zip -d .
python -m pytest tests/test_t02_moment_training.py -v
```

## HPC one-batch smoke test

```bash
export DATASET_ROOT=/absolute/path/to/pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5_normalized
PYTHONPATH="$PWD" python tests/smoke_t02_4_real_batch.py \
    --config config/config_moment_example.yaml --workers 0
```

Use `--workers 0` on a login node warning that only one CPU is assigned. Run training epochs on scheduled compute resources.

## Design decisions

Moment mode uses αs:7 and Q0:5 cross-entropy. `accuracy` means **joint exact-match** (both parameters correct for a sample). Per-head accuracy and macro-F1 are reported separately. Early stopping uses validation joint exact-match. The legacy three-head train/evaluation/logger paths remain unchanged by default. No energy-loss metrics are fabricated. The test-set loader is **not evaluated during training**. The ViT implementation remains regular timm ViT, not ViT-CoMer.

## Limitations / future work

- Do not mistake a one-batch smoke test for convergence testing or proof of legacy regression compatibility.
- Scheduler `OneCycleLR` requires per-batch stepping and is not supported by this epoch-level moment loop; use ReduceLROnPlateau or other epoch-level schedulers.
- The `run_training_with_config` path retains some preexisting assumptions and should be integration-tested on your exact checkout.
- Training with shared source jets across aggregate-disjoint partitions must be disclosed in the experimental methodology.
