# T04-06 Stage 3F.2C2 — Independent v2 scheduler integration

## Scope

**Adds files only.** This stage does not edit legacy registry, legacy job files, existing runs, or the generated v2 plan. The v2 plan from 3F.2C1 must already be verified. This patch contains:

- `tools/t04_06_v2_full_horizon.py`: bounded, held-only launcher pinned to v2 plan, v2 protocol, and v2 Slurm script.
- `tools/t04_06_v2_reconcile.py`: read-only default Slurm/checkpoint reconciliation, with terminal status only at configured epoch count; a legacy early-stop result cannot become v2 completion.
- `slurm/t04_06_v2_single.slurm`: separate logs, 2 CPUs, A100 MIG, 12GB, 12h, fixed CPU thread limits.
- `tests/test_t04_06_v2_orchestration.py`: nine CPU-only regression tests.

The existing `tools/t04_06_registry.py` must already contain v2 kind dispatch from Phase 3F.2C1; this patch does not rewrite it.

## Install and test

```
cd ~/ml-jet-ext
unzip -o T04-06_stage3f_phase2c2_v2_scheduler.zip -d .
python -m pytest -q tests/test_t04_06_v2_orchestration.py
python -m pytest -q tests/
python -m tools.t04_06_v2_matrix verify
python -m tools.t04_06_registry verify
git diff --check
bash -n slurm/t04_06_v2_single.slurm
```

## Bootstrap v2 operational registry (only once)

The immutable v2 planning registry remains `training_output/T04/t04_06_v2/registry.json`. Bootstrap adds a *different* mutable registry. Do not use defaults of the legacy registry CLI.

```
python -m tools.t04_06_registry \
  --runtime training_output/T04/t04_06_v2/registry.runtime.json \
  bootstrap --repo "$PWD" \
  --plan training_output/T04/t04_06_v2/registry.json \
  --commit "$(git rev-parse HEAD)"

python -m tools.t04_06_registry \
  --runtime training_output/T04/t04_06_v2/registry.runtime.json verify

python -m tools.t04_06_v2_full_horizon status
python -m tools.t04_06_v2_full_horizon plan --scope all --count 4 --segment-epochs 24
```

Expected v2 states: 120 planned; 0 active, 0 completed. Plan does not mutate any artifact. **Stop after this read-only plan and share results before preparing/submitting jobs.**

## Later guarded sequence (not for this validation stage)

Once the v2 launcher and data preflight are accepted, use `prepare-wave --scope all --count 4 --segment-epochs 24`, then inspect each frozen run and its manifest. `submit-held-wave` registers actual Slurm jobs but does not release them. This command refuses submission while any legacy production job remains active. Actual release requires manual `scontrol release JOBID` after verifying held job owner, CPU/GPU resources, exact immutable run, mode, target epoch, and intent. No automatic release or retry.

After terminal Slurm states, use `python -m tools.t04_06_v2_reconcile inspect` and verify checkpoints, then `apply` only for expected correct decisions. Intermediate epoch < configured max is `ready_to_resume`; epoch == max is `completed`; any early-stopped v2 result is blocked. For the initial 80-epoch plan, segments may be 1–24,25–48,49–72,73–80. The budget comes from the frozen config; other configured budgets also work.

### Protocol scope

The new matrix is fresh-start only, with independent v2 outputs. No migration of legacy checkpoints. Every v2 immutable source snapshot captures the updated `train_utils` modules, including `t04_training_policy.py` and `t04_training_termination.py`. The first disposable GPU smoke test before large-scale production is still recommended to confirm the snapshot worker and CLI execution end to end.

### Scientific caveat

The dataset's `aggregate_disjoint` protocol must not be described as `source_disjoint`. Report per-seed mean/std of validation and separately controlled test results; don't use test metrics to choose checkpoints.
