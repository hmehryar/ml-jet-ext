# T04-10-A5 / issue #31: Top-k and physically ordered evaluation

This patch extends **only the T04-10 standalone single-head epoch trainer**.
It does not change T04-07, existing joint models, optimizer, loss function,
checkpoint selection (`val.macro_f1`), scheduling, or Slurm submission.

## New validation/training history fields

- `top1_accuracy`, `top2_accuracy`, `top3_accuracy`
- `within_1_class_accuracy`, `within_2_class_accuracy`
- Existing `accuracy`, `macro_f1`, `physical_mae`, `physical_rmse`

`accuracy` remains equal to `top1_accuracy` (checked in code). Top-k is a
**ranked confidence set**, not physical proximity. Ordinal tolerances operate
on top-1 class-index distance (<=1 / <=2). Class values follow the sorted,
explicit 7-class alpha grid (0.20 to 0.50 by 0.05) and 5-class q0 grid
(1.0 to 3.0 by 0.5). Scores are sorted descending with deterministic tie
break to the lower class index. Floating-point logits are cast to FP32 to
compute rankings; BF16 CUDA models are compatible.

Both training and validation epoch result dictionaries acquire the new metrics;
this is supplementary logging only. Historical checkpoints do not acquire
retroactive metrics. The trained model does not receive an unused second head.

## Optional validation-only per-sample output

With a validation **non-shuffled, sequential** DataLoader whose dataset order
exactly matches stable aggregate IDs in `sample_ids`:

```python
from train_utils.t04_10_training import run_single_head_epoch
metrics = run_single_head_epoch(
    val_loader, model, task="alpha_only", device="cuda", precision="bf16",
    prediction_export_path="/dedicated/t04_10_single_head/run/validation_predictions.json",
    sample_ids=[str(path) for path in val_loader.dataset.paths],
    export_metadata={
        "run_id": "frozen-run-id", "architecture": "clip", "strategy": "S4",
        "moments": [1, 2, 3], "checkpoint_sha256": "...sha256...",
        "split": "val", "selection_policy": "best_val_macro_f1",
    },
)
```

**Only** pass sample IDs from the validation dataset in exactly its iteration
order. If sampler order differs, IDs would be incorrectly associated with
predictions. The exporter refuses training, test-split exports, duplicate IDs,
missing provenance metadata, and existing export files. It writes a new JSON
file via a no-clobber hardlink, preserving existing run artifacts. The caller
should load the selected `best.pth` before final validation export if the
artifact is intended to describe the selected checkpoint. These optional
exports are **not automatically enabled by the T04-10 training runner** to
avoid unbounded per-epoch prediction storage and preserve the immutable run
configuration. Predeclared evaluation-only post-processing may enable them later.

For dual-head use, instantiate separate `ClassificationMetrics` accumulators
for each head's logits and ground truth; the utility is model-agnostic.

## Validation

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/test_t04_10_metrics.py tests/test_t04_10_single_head.py
PYTHONPATH=. python -m pytest -q tests/test_t04_10_stage2_preflight.py tests/test_t04_10_stage2_matrix.py tests/test_t04_07_losses.py tests/test_t04_legacy_compat.py
```

CPU tests pass in the patch build environment; CUDA tests require an allocated
GPU and have **not** yet been validated with real CLIP/SigLIP checkpoints.

## Deferred to T04-09

Cross-architecture parity comparisons, loss ablation analysis, and final
production-test reporting will be conducted separately. Do not use `top5` as
a primary Q0 metric: there are only five Q0 classes.
