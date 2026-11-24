# T04-10 Stage 5C — T04-07 Slurm logging parity

Scope: T04-10 only. No T04-07 modifications, no automatic submission or release.

## Per-epoch format

After atomic checkpoint/history commits, print:

- `[T04-04] epoch telemetry: {JSON}` using the existing `train_utils.t04_telemetry` collector and its original machine-readable field names.
- `[T04-10] Epoch N / 80: val alpha|q0 macro-F1 X (best Y @ K)`.

Training run boundaries: `T04 manifest verification: PASS`, `EXECUTING fresh|resume ...`, `T04-10 session completed: ...` and `SESSION_COMPLETED=...`.

Checkpoint selection remains by **single-head validation macro-F1**; this is different from T04-07 joint accuracy. No checkpoint, training-history, model, optimizer or RNG formats are modified. `PYTHONUNBUFFERED=1` in the new Slurm launcher ensures immediate output. The existing `PYTHONDONTWRITEBYTECODE=1` safeguard must remain.

Telemetry `elapsed_seconds` is the train+validation epoch elapsed time as measured by the existing collector. Throughput divides the number of *training* samples by that full epoch elapsed time, consistent with the collector definition. The optimizer-step count reflects one successful `optimizer.step()` call per training batch; no gradient scaler is used. Peaks include the train/validation interval. Telemetry is printed after checkpoints, and is not separately persisted beyond the Slurm log.

## Safe migration for three previously held jobs

The original held Slurm jobs `41841576`, `41841577`, `41841578` contain the **old spooled script** and point to immutable prepared snapshots from commit `9ab085e`. They will not pick up the new telemetry just because files in the Git checkout change.

1. Verify pytest and Bash syntax, stage ONLY T04-10 files, commit/push the new code.
2. Prepare fresh replacement runs for the 3 tasks with new run IDs, e.g. `t04_10_clip_q0_only_seed42_telemetry_v1`, `t04_10_siglip_alpha_only_seed42_telemetry_v1`, and `t04_10_siglip_q0_only_seed42_telemetry_v1`. Do not edit existing snapshots.
3. Verify each newly prepared manifest/source digest, preview and submit `--hold`. Record replacement job IDs.
4. Once replacements are held and verified, cancel ONLY the 3 old T04-10 jobs by their **exact IDs**. Do not use name-based or account-wide cancellation.
5. Release one replacement to validate log format, telemetry, checkpoints and resource behavior before releasing the others.

Leave currently running `41843226` unchanged. The original T04-07 jobs, paths, configs and source remain untouched.
