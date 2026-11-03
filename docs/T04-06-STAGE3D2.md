# T04-06 Stage 3D.2 — successful partial segment + held resume

Pre-existing full regression: **211 passed** on HPC; this patch adds four focused tests.

**Preservation:** Do not regenerate the 120 configs or bootstrap the operational registry. This patch changes only the registry/reconciliation modules and adds `tools/t04_06_resume_gate.py`, tests and documentation. The new state is `ready_to_resume` with the existing schema version. Historical attempts and event history remain.

## CPU-only validation

```
python -m pytest -q tests/test_t04_06_stage3d2.py tests/test_t04_06_reconcile.py tests/test_t04_06_registry.py
python -m pytest -q tests/
python -m tools.t04_06_reconcile inspect
python -m tools.t04_06_registry verify
```

Expected inspection for job 40941279: `action=ready_to_resume`, epoch=12. The `inspect` command is read-only.

## Explicit one-time reconciliation

Only after inspecting the read-only report and confirming Slurm job `40941279` really is terminal `COMPLETED 0:0` and session-001 `result.json` reports epoch 12 / no early-stop:

```
python -m tools.t04_06_reconcile apply
python -m tools.t04_06_registry verify
python -m tools.t04_06_resume_gate inspect
```

Expected state: **119 planned, 1 ready_to_resume**. The reconciler advances `submitted -> running -> ready_to_resume` with audited checkpoint SHA/session, and does not submit Slurm jobs.

The new resume inspector independently checks Slurm accounting, immutable provenance, checkpoint SHA, session, epoch CSV and serialized optimizer/scheduler/RNG state. It never changes any file.

## Live continuation, only after inspected evidence passes

`python -m tools.t04_06_resume_gate submit-held` submits **one held** job for epoch 13 using the *same* frozen run, `--mode resume`, and creates a distinct durable intent. It records a new Slurm ID and parent checkpoint hash in the existing operational registry. This is an explicit action, not part of a dry-run. **Do not release it automatically.** Verify Slurm ownership/exports and registry first, then deliberately issue `scontrol release <jobid>` if approved.

If `sbatch` or registry update fails, the intent blocks automatic retries. Manually reconcile an uncertain Slurm submission; never remove the intent just to retry.

**Limitations:** The existing checkpoint restoration code must still be validated via the epoch-13 Slurm acceptance test. The CLI does not automatically finalize interrupted sessions, requeue jobs, submit all 120 configs, or interpret a 12-epoch partial segment as an 80-epoch completed experiment. The existing 2-DataLoader-worker warning with a one-CPU allocation persists because configs are hash-bound.
