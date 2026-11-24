# T04-06 Stage 3F.2C2.1 — short segment worker compatibility

This patch allows fixed_budget_v2 to execute 3-epoch segments when min_epochs=12, leaving legacy early-stop behavior unchanged. Requires a new immutable run snapshot. The quarantine tool only moves the four known unsubmitted v2 prepared runs after registry, run-manifest, no-attempt, no-session and no-submission-intent validation. No Slurm jobs are submitted or canceled.

1. Apply worker patch to top-level only.
2. Run all tests and registry verification.
3. Check then apply quarantine, then prepare-wave again for the same four IDs/three epochs.
4. Verify new immutable snapshots contain the new check.
5. Held Slurm submission requires a separate manual approval step.
