# Stage 3: eight immutable E5 smoke runs

This patch DOES NOT submit jobs. Use a fresh `feat/t04-05-mig-feasibility` branch based on T04-04.

1. `python -m tools.t04_05_smoke prepare --manifest "$(pwd)/training_output/T04/feasibility/t04-05/smoke_manifest.json"`
2. `python -m tools.t04_05_smoke verify --manifest "$MANIFEST"`
3. Submit a single fresh array with `--array=0-7%4`, `T04_05_MODE=fresh`, `T04_05_TARGET=1`.
4. After every fresh task succeeds and has valid epoch-1 checkpoint, submit a single resume array with `T04_05_MODE=resume`, `T04_05_TARGET=2`.
5. `python -m tools.t04_05_smoke verify --manifest "$MANIFEST"`, then run `python -m tools.t04_04_verify_pilot --run "$RUN"` for each run and inspect validation metrics/telemetry. Do not close issue until all eight pass.

The manifest is exclusive-created, and the preparation tool never modifies existing prepared source snapshots. If preparation aborts halfway, existing run directories can be validated and reused, but the run configuration must match exactly. Do not edit configurations after preparing them. Each array task has independent snapshot/session histories; no cross-task mutable state.

Important: S1–S4 and CLIP/SigLIP may have different trainable parameter counts and memory/time profiles. Batch 8 / accumulation 1 is a feasibility-based smoke choice; it differs from the earlier effective batch 4 pilots. Slurm array `%4` caps the array, not unrelated HPC jobs.
