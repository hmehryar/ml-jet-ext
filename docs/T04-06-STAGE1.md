# T04-06 Stage 1 — Deterministic factorial planning

Issue: https://github.com/hmehryar/ml-jet-ext/issues/25

This is a CPU-only **planning** patch. It does not reserve immutable runs or submit Slurm jobs.

Matrix: 2 models × 4 strategies × 5 cumulative moment sets × 3 seeds = **120**.
Seeds: 42, 123, 2026 (proposed; choose/approve before generating). Base templates: T04-05 validated smoke YAMLs. Strategy: S1–S4; inputs E1–E5. BF16, AdamW, batch 8, accumulation 1 as initial profile only. Training horizon proposal: epochs=80, min_epochs=12, patience=16; not yet qualified for full production. Exact matching checkpoint revisions and split paths inherited from the smoke templates. `aggregate_disjoint` with source overlap disclosed, **not source-independent**.

Commands:

```bash
python -m pytest -q tests/test_t04_06_matrix.py
python -m tools.t04_06_matrix dry-run | tail -n 6
python -m tools.t04_06_matrix generate
python -m tools.t04_06_matrix verify
```

- `generate` creates 120 deterministic YAML files in `experiments/t04/config/t04_06/` and an immutable *plan* JSON under the ignored `training_output/T04/t04_06/registry.json`.
- Plan checks all file paths before writing and refuses existing configs/registry/output paths; never overwrite or regenerate an existing plan blindly.
- Each row includes the source-template hash, generated YAML hash, expected unique run path, identity, and planned state.
- `verify` rederives the full matrix and verifies all configuration hashes. It intentionally fails once a production run directory has been reserved; use a future operational registry validator for that state.
- Stage 2 must implement transactional prepare, durable job registry transitions, Slurm array job mapping, actual MIG availability/QoS preflight, and resume/requeue handling before any submission. `dry-run` currently cannot provide measured per-configuration resource estimates, so do **not** label it the final acceptance dry-run.
- Keep T04-05 smoke runs and artifacts separate. Never read held-out test targets for training/selection; use validation only and pin the existing hashed split inputs.

**Run count caveat:** T04-05 completed eight E5 *smoke* experiments using seed 42; these are not replacements for T04-06 production jobs. T04-06 includes 24 E5 primary jobs across three seeds, and 96 E1–E4 primary jobs.
