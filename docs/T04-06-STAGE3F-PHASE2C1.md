# Stage 3F.2C1: Independent fixed-budget v2 matrix

Scope: New **fresh-start** 120-experiment plan only, without checkpoint migration.

## Design

- 2 models × 4 strategies × 5 cumulative moment sets × 3 seeds = 120.
- Fresh initialization from each strategy's pinned checkpoint source; **no v1 weights, optimizers, sessions, or run IDs reused**.
- `training_protocol: fixed_budget_v2` plus `training_protocol_hash` bound to `epochs`.
- Epoch budget defaults to 80 and can be changed explicitly at plan creation.
- The already tested training loop disables early stopping for v2, keeps LR scheduler and best checkpoint.
- New paths: `experiments/t04/config/t04_06_v2/`, `training_output/T04/t04_06_v2/registry.json`, `training_output/T04/t04_06_v2/registry.runtime.json`, and `training_output/T04/t04_06_v2_production/`.
- New v2 runs have `-v2` suffix. Legacy artifacts are not modified.
- `tools.t04_06_registry` bootstrap accepts either verified legacy or v2 plan, but remains a separate immutable runtime file for each.

## Install and validate

```bash
unzip -o T04-06_stage3f_phase2c1_independent_matrix.zip -d .
python -m tools.t04_06_patch_v2_schema --check
python -m tools.t04_06_patch_v2_schema --apply
python -m pytest -q tests/test_t04_06_v2_matrix.py
python -m pytest -q tests/
git diff --check
python -m tools.t04_06_registry verify  # legacy remains unchanged
python -m tools.t04_06_v2_matrix dry-run --epochs 80
```

No files or jobs are created by dry-run. Check user-provided legacy Slurm jobs independently.

## Generate the independent v2 plan (after dry-run review)

```bash
python -m tools.t04_06_v2_matrix generate --epochs 80
python -m tools.t04_06_v2_matrix verify
```

**DO NOT bootstrap or submit yet**. The Stage 3F full-horizon launcher currently contains hard-coded legacy checks (`epochs == 80`, `min_epochs == 12`, `patience == 16`), and reconciliation checks legacy registry defaults. The next Stage 3F.2C2 patch adds v2-aware launcher/reconciler selection and GPU acceptance tests before any live submission.

### Correctness cautions

- Dataset split protocol is `aggregate_disjoint`; do not claim source-jet independence.
- v2 must be started fresh. It never resumes an early-stopped v1 run.
- Source snapshots taken at `prepare()` must include policy modules. Preflight schema patch makes the protocol keys appear in frozen `config.resolved.json`.
- For Slurm v2, assert that training worker executes from new immutable snapshot and enforces the new policy hash.
- Legacy active jobs must finish under original code. Nothing here changes them.
