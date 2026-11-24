# T04-06 Stage 3B — conservative Slurm reconciliation

Read-only `inspect` is the first acceptance step. `apply` is an explicit opt-in; never call automatically from an array worker or cron before additional recovery work is validated.

- `sacct -n -P --format=JobIDRaw,State,ExitCode` is parsed only for exact array task IDs; `.batch`/`.extern` are excluded.
- Unobserved Slurm IDs and unexpected states are **unknown**, not failed.
- Completed Slurm jobs require immutable manifest verification, SHA-verified latest checkpoint, consecutive metrics CSV rows and a completed session result. Absent evidence is not promoted to completed.
- `submitted/resumed -> running -> completed/failed` follows the existing Stage-2 transition API. Stage 3B does not submit or reserve training jobs.
- `failed` is a scheduler attempt outcome, **not** authorization to automatically resume; eligibility and an explicit resume submission are Stage 3C.
- Existing 120 planned entries remain untouched. T04-05 smoke runs remain separate.
- Stage 3C must additionally verify actual scheduler ownership before transitioning on race-prone `apply`, and handle `sacct` eventual consistency, `COMPLETING`, interrupted sessions, and checkpoint compatibility.

## HPC acceptance

```bash
unzip -o T04-06_stage3b_reconciliation.zip -d .
python -m pytest -q tests/test_t04_06_reconcile.py
python -m pytest -q tests/
python -m tools.t04_06_reconcile inspect
python -m tools.t04_06_registry verify
```

Initially, the 120 jobs are still `planned`; `inspect` should print `RECONCILED 0 active entries; mode=inspect`. No files are written unless `--output` explicitly names a new report path (exclusive creation).

Avoid `apply` until full Stage 3C scheduler recovery, ownership checks, and end-to-end acceptance are ready.
