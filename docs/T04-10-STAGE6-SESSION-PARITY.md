# T04-10 Stage 6: T04-07 session-based production parity

Production implementation for the two single-head tasks `alpha_only` (7 classes)
and `q0_only` (5 classes), retaining the frozen 80-epoch, validation-only
macro-F1 selection protocol. **Does not touch T04-07**.

## Shared T04-07 lifecycle contract

The session helper `train_utils/t04_10_sessions.py` is an isolated adapter of
`train_utils/t04_sessions.py` (T04-02/T04-07 production engine), replacing the
preflight verifier with T04-10's source-audited verifier. It follows the same:

- `experiment_lock` (nonblocking `flock` throughout execution),
- append-only session numbering (`session-001`, `session-002`, ...),
- reserved/running/completed/interrupted session manifests,
- atomic SHA-256 `latest_checkpoint.json` pointer,
- session-local `checkpoints/` and `telemetry/`,
- resume parent checkpoint provenance and strictly increasing epochs,
- latest/best checkpoint retention for each session,
- `[T04-04] epoch telemetry` and per-epoch progress in Slurm `.out`,
- `EXECUTING ...` and `SESSION_COMPLETED=...` log markers.

T04-10-specific checkpoint format is schema=2; it stores the *single active
head*, optimizer/scheduler/RNG state, macro-F1 best epoch and authoritative
history. It NEVER emits an inactive head or joint accuracy. Best model selection
remains by task macro-F1; the scheduler monitors validation loss. Existing
T04-10 flat `training/` checkpoints have schema=1 and are intentionally **not**
accepted by the Stage 6 resume pathway.

## Experiment directory

```text
<t04_10_root>/<model>/<task>/E<k>/seed-<seed>/<run_id>/
  config.resolved.json
  run_manifest.json
  source/                          # immutable snapshot
  .session-execution.lock         # advisory lock, safe to persist
  latest_checkpoint.json          # SHA-256 validated rolling pointer
  history.json                    # global, checkpoint-recoverable history
  single_head_epoch_metrics.csv   # compact global metrics
  sessions/
    session-001/
      session_manifest.json
      checkpoints/
        epoch_000080.pth           # latest (or last completed epoch)
        best_epoch_0000xx.pth      # immutable hard link; retained if best
      telemetry/
        epoch_000001.json
        ...
      single_head_epoch_metrics.csv
    session-002/                   # only after interrupted prior session
      session_manifest.json
      checkpoints/
      telemetry/
```

## Production guard and rollout

**Never run `--apply` before passing CPU tests and preparing a NEW run ID.**
Old seed-42 `*_pilot_v1` and `*_telemetry_v1` runs remain historical evidence;
Stage 6 guard refuses their flat `training/` directories and source snapshots.

1. Install patch on branch `feat/t04-10-single-head-pilot`, commit it, push.
2. Run `python -m pytest -q tests/test_t04_10_stage6_sessions.py tests/test_t04_10_stage5a_guard.py tests/test_t04_sessions.py tests/test_t04_session_training.py`.
3. For one selected seed-42 config, run `python -m train_utils.t04_10_preflight --config <path> --prepare-only --run-id <new-stage6-id>`.
4. `python -m train_utils.t04_10_preflight --verify <new-run>`.
5. `bash tools/t04_10_submit_held.sh <new-run> fresh` (preview) then `... --apply` (held submission).
6. Verify Slurm `NumCPUs=1, mem=12G, gres/gpu:nvidia_a100_80gb_pcie_1g.10gb=1` and `JobHeldUser`, then release exactly that job.
7. Check `session-001` and its first telemetry and checkpoint before submitting other pilots.

**Fixed Slurm resources:** 1 CPU, one A100 1g.10gb MIG, 12 GiB host RAM,
12 hours, `ecscp/express`. The frozen configs currently request two DataLoader
workers; Slurm may warn that this oversubscribes the single CPU. Do not silently
change immutable configs during this migration.

## Resume and recovery

A previously published checkpoint is verified by SHA-256; if a session was
terminated by `SIGTERM`/`SIGKILL`, its manifest may remain `running`. On an
explicit `resume` operation (under experiment lock), incomplete earlier sessions
are marked `interrupted` and a new session is reserved. Previous session
checkpoints/telemetry are never overwritten. Global CSV/JSON history is
reconciled from authoritative checkpoint state.

Only unfinished fixed-budget epochs can resume; no extension beyond the frozen
80-epoch training budget is permitted. A failure before *any* checkpoint was
published requires a fresh experiment identity. A failure after publishing
Epoch 80 but before session completion requires manual integrity reconciliation
rather than launching more training. This design remains intentionally fail-closed.

## Safety

No T04-07 code, jobs, output paths, or manifests are edited. No legacy
T04-10 run directories are deleted or migrated in place. Test data remains
excluded from validation checkpoint selection. The split's documented
cross-partition source overlap must remain disclosed (103,221 reported
occurrences for the referenced dataset audit).
