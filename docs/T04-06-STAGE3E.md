# Stage 3E — bounded production scheduling: seven E5 seed-42 fresh runs

The already validated `clip_s2_e5_seed42` experiment stays **ready_to_resume** at epoch 13. Do not regenerate or edit the hashed YAMLs. Stage 3E schedules the seven untouched E5 seed-42 identities only; **it does not yet schedule all 120 factorial experiments**.

## Acceptance commands (read-only)

```bash
cd ~/ml-jet-ext
unzip -o T04-06_stage3e_bounded_scheduler.zip -d .
python -m pytest -q tests/test_t04_06_production.py
python -m pytest -q tests/
python -m tools.t04_06_registry verify
python -m tools.t04_06_production plan --count 3
python -m tools.t04_06_production status
bash -n slurm/t04_06_single.slurm
```

## Gated live operation — only after inspection and explicit approval

1. `python -m tools.t04_06_production prepare-wave --count 3` prepares three immutable runs (does not submit).
2. `python -m tools.t04_06_production submit-held-wave --count 3` submits **three held** jobs, each using the existing `slurm/t04_06_single.slurm`, `mode=fresh`, `target=min_epochs` and durable per-experiment submission fencing. They remain **held**.
3. Inspect the registry and `scontrol show job JOB_ID` for **each** job, verify owner/one CPU/12GB/MIG, and explicitly release only authorized jobs using `scontrol release JOB_ID`.
4. After each Slurm job completes, use `python -m tools.t04_06_reconcile inspect`, audit source/checkpoint/session/CSV/telemetry, then `python -m tools.t04_06_reconcile apply` for verified partial segments.
5. **Only after all three are ready_to_resume (or fully completed) and sacct records COMPLETED 0:0**, run `python -m tools.t04_06_production plan --count 4` and repeat steps 1–4 with four jobs.

The stage is intentionally conservative: **it does not automatically release held jobs, retry failures, transition registry states, or resume sessions**. `stage3e.wave.lock` serializes one launcher invocation across processes; original per-experiment intent fencing is retained. A failed or uncertain `sbatch` attempt must be manually reconciled, not retried. A partial wave is not automatically rolled back; remaining held jobs require manual review. The existing Stage 3D.2 epoch-13 run and its provenance are unchanged.

## Known limits

- This is bounded sequential-wave launch orchestration, not a global Slurm capacity scheduler for all 120 experiments.
- `num_workers=2` in the frozen configs still produces an oversubscription warning for one core; changing it requires a new versioned plan and must not modify current immutable runs.
- The previous Stage 3E launcher designed for all eight tasks is incompatible with CLIP S2 already being resumed; use this seven-run controller instead.
- The 80-epoch study requires additional independently audited continuation jobs. A successful minimum-epoch job is **ready_to_resume**, not a completed experiment.
