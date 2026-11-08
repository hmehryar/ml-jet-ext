# T04-10 — Stage 1: task-isolated training checkpoint

Issue: https://github.com/hmehryar/ml-jet-ext/issues/30

This is an **additive patch** for a new branch from `feat/t04-07-loss-ablations`.
No T04-07 configuration, checkpoint, registry, submission script, trainer, or
existing model source is modified.

## Implemented

- `alpha_only` (7 logits) and `q0_only` (5 logits) with exactly one head.
- CLIP/SigLIP vision encoder support with the existing S1–S4 trainability
  adapter, 1x1 input projection, and bilinear interpolation.
- Provisional ViT-Tiny single-head wrapper, pending full architecture parity.
- Task-isolated CE loss, accuracy, macro-F1, physical MAE and RMSE, confusion
  matrix, BF16 autocast, task-safe AdamW optimizer groups.
- Fixed-budget epoch driver with active-task validation macro-F1 checkpoint
  selection, atomic best/last saves, RNG/optimizer/scheduler restoration, and
  config-hash verification on resume.
- No joint accuracy or inactive-target result is fabricated.

## Stage 1 limitations (not production-ready)

- Does not integrate with T04-01 immutable prepared-run manifests or T04-06 HPC
  registry/slurm orchestration. Those need a dedicated task-aware T04-10 adapter.
- The fixed-budget driver requires a fresh unique output dir and supplied
  loaders; callers must provide stable epoch-wise sampler seeding for shuffled
  inputs, plus config/dataset provenance verification.
- CUDA/BF16 tests are automatically skipped on CPU-only hosts; must run on GPU
  before any production submission.
- No HF checkpoint weights are fetched during CPU tests; tests use synthetic
  backbones. Actual pinned checkpoint, 35-configuration dataset and HPC smoke
  require the research environment.
- ViT-Tiny historical transfer-learning run differs from CLIP/SigLIP training
  strategies and budget. **Do not infer transformer architecture rankings** from
  this pilot until a distinct ViT parity milestone implements E1-E5 × S1-S4,
  matching CE and ordinal/physical loss ablations, equal evaluation protocol,
  and matched multi-head controls.

## Apply on an authorized workstation

```bash
git fetch origin
git switch -c feat/t04-10-single-head-pilot origin/feat/t04-07-loss-ablations
# Copy models/, train_utils/, tests/, docs/ from this patch archive into repo root.
PYTHONPATH=. pytest -q tests/test_t04_10_single_head.py
# Re-run on A100/L40S/H200 to pass the two CUDA BF16 tests.
git add models/t04_10_single_head.py train_utils/t04_10_*.py tests/test_t04_10_single_head.py docs/T04-10-STAGE1.md
git commit -m 'feat(t04-10): add isolated single-head training and CPU/CUDA gates'
```

## Next checkpoint

Wire a strict task-aware prepared-run preflight and snapshot runner, dataset
split hashes and output namespace, selected-head epoch metrics, immutable
provenance, and held-only Slurm submissions. The 18 production configurations
remain UNFROZEN pending validation-only baseline audit.
