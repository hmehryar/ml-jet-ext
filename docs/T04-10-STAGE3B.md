# T04-10 Stage 3B — real A100 BF16 2-epoch fresh/resume smoke

Parent issue [#30](https://github.com/hmehryar/ml-jet-ext/issues/30), metrics subtask [#31](https://github.com/hmehryar/ml-jet-ext/issues/31). **Never launch T04-07 jobs from this test.**

## Validation scope

- CLIP and SigLIP pretrained HF vision checkpoints, pinned revisions, strategy S3, E3 moments, seed 42, both `alpha_only` and `q0_only`: 4 cases.
- Exactly six train samples and six validation samples per case, batch size 1, data-loader workers 0. This is a functional smoke, **not** a statistically meaningful experiment.
- `fit_fixed_budget` fresh to epoch 1; verify `last.pth`, `best.pth`, optimizer state, scheduler state, RNG state, task and configuration hash, complete history.
- Reinitialize backbone/model/optimizer/scheduler, deliberately perturb RNG, then resume with the real checkpoint to epoch 2. Validate advancing optimizer step, preserved first-epoch history, full second-epoch metrics.
- Load best checkpoint and repeat its validation evaluation; verify top-1/2/3, macro-F1, ±1/±2 tolerance, physical MAE/RMSE, and optional per-sample validation artifact against checkpoint history.
- **Never execute test-set inference.** Preflight reads test *split metadata and hashes* as part of provenance verification, but model training/evaluation only reads `splits/train.csv` and `splits/val.csv`.
- A failed source-overlap audit is explicitly disclosed, not treated as a source-disjoint split. **Do not make source-independent generalization claims** if overlaps remain.
- All logs, checkpoints, outputs under `/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3b_resume/`, with a separate `job_$SLURM_JOB_ID/` directory per job.

## Installation and execution

From the repo root, in your `pytorch` conda environment and on branch `feat/t04-10-single-head-pilot`:

```bash
mv T04-10-stage3b-resume-validation.zip /tmp/
mkdir -p /tmp/t04_10_stage3b
unzip /tmp/T04-10-stage3b-resume-validation.zip -d /tmp/t04_10_stage3b
bash /tmp/t04_10_stage3b/install_t04_10_stage3b.sh "$PWD"

PYTHONPATH=. python -m pytest -q \
  tests/test_t04_10_stage3b_resume.py tests/test_t04_10_metrics.py \
  tests/test_t04_10_single_head.py tests/test_t04_10_stage2_preflight.py \
  tests/test_t04_10_stage2_matrix.py tests/test_t04_07_losses.py \
  tests/test_t04_legacy_compat.py

mkdir -p /rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3b_resume/logs
bash -n slurm/t04_10_stage3b_resume.slurm
sbatch slurm/t04_10_stage3b_resume.slurm
```

Precondition: Stage 3A generated four S3/E3 configs at `experiments/t04/config/t04_10_stage3_smoke/t04_10_{clip,siglip}_{alpha_only,q0_only}_seed42.yml` and their preflights passed. This script never changes or generates those configs.

To watch job `JOBID`:

```bash
ROOT=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3b_resume
squeue -j JOBID
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
tail -n 100 "$ROOT/logs/T04-10-RESUME_JOBID.out"
cat "$ROOT/logs/T04-10-RESUME_JOBID.err"
find "$ROOT/job_JOBID" -name summary.json -print
```

Expected terminal marker: `STAGE3B_ALL_FOUR_PASS`, four `STAGE3B_RESULT` summaries, Slurm `COMPLETED` and exit `0:0`. Avoid rerunning a job ID's output path; each new Slurm job gets a fresh namespace.

## Boundaries and scientific interpretation

This tests checkpoint *structural* integrity and a real CUDA resume, not bitwise equivalence between uninterrupted and resumed GPU trajectories. The CPU Stage 1 suite already verifies exact resume under its deterministic fixture. CUDA bitwise comparison requires a separate controlled uninterrupted reference with pinned deterministic kernels, worker state, CUDA/PyTorch versions, and full RNG protocol.

S3/E3 configurations are **integration controls**, not validation-selected winning baselines. Before freezing T04-10 pilot configs, rank candidate CE **dual-head** models by a predeclared validation metric, retain architecture-specific selections, ensure the source-overlap limitation is acknowledged, and do not use test data for selection. The six ViT runs remain blocked pending the separate ViT S1–S4 parity implementation.

The committed `slurm/t04_10_single_head_held.slurm` requires a separate pre-production correction: use `--gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1` and `/rs` logs. This package does not modify the production launcher.
