# T04-06 Stage 3F — Phase 1: 120-experiment full-horizon wave launcher

## Scope

Build on the **existing** 120-entry operational registry. **Never bootstrap or
regenerate** the registry. Eight E5 seed-42 experiments are currently
`ready_to_resume` at epochs 12 (seven) / 13 (CLIP S2). Another 112 are
`planned`. No run is terminal-completed yet.

The new `tools/t04_06_full_horizon.py` schedules wave sizes 1..4. It prefers
`ready_to_resume` runs, then never-started runs. It handles both `fresh` and
`resume` (with matching frozen config, latest pointer, last-attempt SHA, session,
independent terminal `sacct`, completed session evidence, optimizer/RNG/scheduler
checkpoint states). New fresh runs are prepared by the existing immutable
preflight module. It does not edit YAML, registry plan, prior sessions, or
previous checkpoints.

**Training policy:** `epochs=80` is the maximum, `min_epochs=12`, `patience=16`.
Default segment size is **24 additional epochs**; the first segment of a new
experiment targets epoch 24. A resumed experiment at epoch 12 targets epoch 36;
at epoch 13, 37. The trainer can early-stop before the segment target when
allowed. After a segment, Slurm `COMPLETED` + verified checkpoints/session
will reconcile to `ready_to_resume`, or `completed` if valid early stopping /
80-epoch maximum was reached. A Slurm `TIMEOUT`, `FAILED`, or an unfinished
session requires manual checkpoint and scheduler audit before any recovery.

**Safety:** Submit commands always use `sbatch --hold`. A durable per-experiment
AND per-parent-epoch fence is written before `sbatch`; no automatic resubmit,
release, or cancellation is performed. Registered active attempts occupy one of
4 slots until the operational registry reconciles, even if they vanish from
`squeue`. `--scope all` enables all 120 (not just E5), subject to the bounded
wave. The Slurm script must declare 2 CPU cores per job, 1 OMP/MKL thread,
12GB and one A100 MIG instance.

## Stage 3F Phase 1 HPC acceptance (read-only first)

```bash
cd ~/ml-jet-ext
unzip -o T04-06_stage3f_phase1_full_horizon.zip -d .
python -m pytest -q tests/test_t04_06_full_horizon.py
python -m pytest -q tests/
git diff --check
python -m tools.t04_06_registry verify
python -m tools.t04_06_full_horizon status
python -m tools.t04_06_full_horizon plan --scope e5 --count 4 --segment-epochs 24
```

Expected: 8 ready-to-resume, 112 planned, no active slots. The plan should
prioritize four checkpointed E5 runs and **must not submit jobs**.

## Controlled training wave (only after read-only acceptance)

```bash
python -m tools.t04_06_full_horizon prepare-wave --scope e5 --count 4 --segment-epochs 24
python -m tools.t04_06_full_horizon submit-held-wave --scope e5 --count 4 --segment-epochs 24
```

The second command **submits held Slurm jobs** and changes the runtime registry.
**Do not run it until explicitly ready for new GPU work.** Verify every
returned Slurm ID and its held state, CPU=2, MIG GPU=1, mem=12G, QoS, job
working directory, `T04_06_MODE`, `T04_06_RUN`, `T04_06_TARGET`, and parent
checkpoint SHA recorded in each new intent. Then manually release the actual
held IDs with `scontrol release <ID>`. There is deliberately no release command
in this Python launcher.

After Slurm terminals appear, inspect **before** applying:

```bash
python -m tools.t04_06_full_horizon reconcile-inspect
python -m tools.t04_06_full_horizon reconcile-apply
python -m tools.t04_06_registry verify
python -m tools.t04_06_full_horizon status
```

Only apply reconciliation after checking all decisions. If any entry reports
`none` or `failed`, investigate individually; do not blindly roll the queue.

Repeat bounded waves. When the initial E5 group is fully terminal-completed,
run with `--scope all` to include other E1–E4 and seed values. This manual
wave cycle is **not** a background worker and does not yet autonomously drain
all 120 jobs. That would need an additional conditional release/retry gate and
monitoring/scheduler quota policy.

## Caveats

- The T04-06 dataset protocol is **aggregate_disjoint** with previously disclosed
  cross-partition source overlap; do not claim source-independent results.
- A new Slurm CPU allocation for already-prepared checkpoint runs changes
  resource utilization, not the hash-bound ML data/configuration.
- Training source code remains frozen within each run directory.
- The existing `t04_06_reconcile` will leave ambiguous Slurm/checkpoint evidence
  untouched. Never rewrite status JSON or delete intents to force progress.
- A job may disappear from `squeue` after finishing. Query `sacct` for history.
- The launcher has no automatic post-submission release or overnight background
  scheduling. Do not assume it autonomously runs all 120 experiments.
