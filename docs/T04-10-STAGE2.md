# T04-10 Stage 2 — isolated preflight, dataset integration, and held HPC execution

**Issue:** #30. **Parent:** Stage 1 single-head classifiers (`abbb811`). **Not a T04-07 replacement.**

## Scope and gates

- Adds independent `train_utils/t04_10_{preflight,runner}.py` modules, standalone real-CUDA smoke, experiment matrix generator, and held Slurm launcher.
- Does **not** change existing T04-07, T04-06 or T03 training implementation, files, run directories, or scheduled jobs.
- Initial runnable architecture support is **CLIP and SigLIP**. The six ViT entries of the 18-run matrix remain **blocked** until a validated ViT training-parity path is implemented in its own task. Historic ViT performance is not a matched comparison.
- Requires an explicit source-ID overlap report matching the train/validation/test CSV SHA-256 hashes. `aggregate_disjoint` is allowed **only with source-overlap disclosed**. `source_disjoint` requires zero overlap.
- No test-set inference is performed by this stage. Hold test until all configurations, heads and checkpoints are selected exclusively from validation results.
- All runs require pinned checkpoint revisions, an independent T04-10 output root, seed in {42, 123, 2026}, E1–E5 cumulative moments, CE, 80-epoch `fixed_budget_v2`, AdamW and BF16/fp32.
- Preflight is CPU-only, fail-closed, records Git commit and source/split/audit hashes. Source and config are copied into each experiment's private `source/` snapshot and SHA-256 verified before the CUDA worker starts.
- Only one training process per run is allowed via exclusive `training.lock`; held Slurm submission requires deliberate release.

## Before installing

The already-pushed T04-10 branch predates the newer T04-07 `4bf039d` commit. On HPC:

```bash
cd ~/ml-jet-ext
# Commit/stash any tracked work first; preserve untracked archive files.
git fetch origin
git switch feat/t04-10-single-head-pilot
git merge --no-ff origin/feat/t04-07-loss-ablations -m "merge: sync T04-07 changes into T04-10 baseline"
```

Then extract the Stage 2 ZIP outside the repo and copy **only** its new files, using the included installer. Do not copy to T04-07 and do not run `git add .`.

## Dry-run and real GPU smoke

Select a validated *CE* dual-head configuration separately for CLIP and SigLIP; do not use T04-07 loss results or the test set to pick a winner. Generate the candidate matrix:

```bash
PYTHONPATH=. python tools/t04_10_generate_matrix.py \
  --clip-baseline experiments/t04/config/t04_06_v2/clip_s4_e3_seed42.yml \
  --siglip-baseline experiments/t04/config/t04_06_v2/siglip_s4_e3_seed42.yml \
  --output experiments/t04/config/t04_10_candidate
```

These are **illustrative candidates**, not measured winners. The generator emits 12 CLIP/SigLIP configs and a plan recording six blocked ViT entries. The output path must not already exist.

Run CPU-only preflight for a chosen candidate, with no CUDA load:

```bash
PYTHONPATH=. python -m train_utils.t04_10_preflight \
 --config experiments/t04/config/t04_10_candidate/t04_10_clip_alpha_only_seed42.yml --check
```

Run real-checkpoint CUDA smoke first on a GPU allocation (do not run it on a CPU-only login node):

```bash
PYTHONPATH=. python tools/t04_10_real_cuda_smoke.py \
 --config experiments/t04/config/t04_10_candidate/t04_10_clip_alpha_only_seed42.yml
```

Repeat for both architectures and both tasks, and for the candidate strategies. Real pretrained weights must be available in the HF cache or reachable by the compute node; these are **not** verified by CPU unit tests.

Prepare a single immutable run after source audit and pilot baseline review:

```bash
PYTHONPATH=. python -m train_utils.t04_10_preflight \
 --config experiments/t04/config/t04_10_candidate/t04_10_clip_alpha_only_seed42.yml --prepare-only
```

Capture the printed prepared-run path and submit a **held** job:

```bash
bash tools/t04_10_submit_held.sh /absolute/path/to/prepared/run fresh
squeue -u "$USER"
# Inspect the job, then *explicitly* release the chosen ID when ready:
# scontrol release JOB_ID
```

`--hold` is mandatory for submissions from this launcher. Check the `--gres`, partition, QoS, walltime, memory settings against the specific WSU allocation before releasing. The Slurm script and logs are T04-10 specific.

## Tests and limits

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_t04_10_single_head.py \
  tests/test_t04_10_stage2_preflight.py \
  tests/test_t04_10_stage2_matrix.py \
  tests/test_t04_07_losses.py \
  tests/test_t04_legacy_compat.py
```

Stage 2 CPU preflight and synthetic Stage 1 tests are not proof of a real CLIP/SigLIP GPU end-to-end training run, nor of bitwise-exact multiworker CUDA restart. Before scheduling production, verify actual CUDA/BF16 smoke, source audit, memory, checkpoint roundtrip/resume and run manifest integrity against the current HPC environment.
