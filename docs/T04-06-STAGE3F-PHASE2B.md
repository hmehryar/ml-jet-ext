# Stage 3F.2B — Configurable fixed-budget protocol

Protocol ID is `fixed_budget_v2`; `epochs` comes from the resolved config, not a hard-coded 80.
The policy digest includes the configured epoch budget: e.g. 80 and 120 epochs produce distinct hashes.
No early stopping in v2; evaluation, best checkpoint, and LR scheduling remain intact.
Legacy behavior remains unchanged when `training_protocol` is omitted.

**This is only the training-core integration patch.** No migration or new job may be submitted until
Phase 3F.2C creates an immutable v2 run with pinned policy hash, config, source snapshot,
checkpoint-parent lineage, and a separate registry. Never mutate the legacy registry or snapshots.
Do not modify or cancel active Slurm jobs.

## Install and verify

    unzip -o T04-06_stage3f_phase2b_configurable_budget.zip -d .
    python -m tools.t04_06_patch_training_policy --check
    python -m tools.t04_06_patch_training_policy --apply
    python -m pytest -q tests/test_t04_training_termination.py tests/test_t04_06_protocol_audit.py
    python -m pytest -q tests/
    git diff --check
    python -m tools.t04_06_registry verify
    python -m tools.t04_06_protocol_audit preview --epochs 80

If `--check` rejects source snippets, do not force it. The old Phase 2A fixed80
module may remain on disk but is unused by the revised policy and audit.
