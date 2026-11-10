# T04-10 Stage 5A — production launcher and isolation gate

Scope: issue #30 only. No T04-07 source, configuration, registry, jobs, or run artifacts are modified. Production T04-10 output root is `/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head`. `slurm_logs/` is a dedicated subdirectory; `sbatch` writes there directly. No symlinks into T04-07 are permitted by run canonical-path verification.

## Safety contract

1. `tools/t04_10_submit_held.sh RUN [fresh|resume]` is **preview-only** by default, validating the prepared manifest and immutable run snapshot but creating neither directories nor Slurm jobs.
2. Only `... --apply` creates exactly one **held** Slurm job. The script calls `sbatch --parsable --hold`, never releases or cancels jobs, and only creates the T04-10 log directory.
3. The Slurm payload reruns the run guard prior to importing model/training dependencies; a manually released stale or corrupted job fails closed.
4. `fresh` refuses an existing `training/` directory; `resume` requires `training/last.pth` and `training/history.json`. Both refuse an existing `training.lock`.
5. The launcher retains `2` CPUs, `12G` RAM, `12h`, `ecscp` / `express` and one A100 1g.10gb MIG; no speculative CPU reduction. Measure DataLoader throughput in Stage 5B/5C before changing this.
6. No test-set evaluation or source-disjointness claim. Known source overlap 103221 is disclosed via immutable run manifest.

## Validation (CPU only)

```bash
PYTHONPATH=. pytest -q tests/test_t04_10_stage5a_guard.py tests/test_t04_10_stage2_preflight.py
bash -n tools/t04_10_submit_held.sh
bash -n slurm/t04_10_single_head_held.slurm
```

## After Stage 5B explicitly prepares a seed-42 run

```bash
# Preview, no scheduler call:
bash tools/t04_10_submit_held.sh /absolute/t04_10/prepared/run fresh

# ONLY after approval: submit one HELD job, not running:
bash tools/t04_10_submit_held.sh /absolute/t04_10/prepared/run fresh --apply

# Inspect the returned job ID. Release manually only when appropriate:
squeue -j JOB_ID
# scontrol release JOB_ID
```

**Do not prepare, submit, or release jobs as part of Stage 5A.** The frozen Stage 4 YAMLs are not prepared run directories; Stage 5B creates the latter with `train_utils.t04_10_preflight --prepare-only` after code is committed.
