# T04-02 Stage 2: Immutable training sessions

Additive, CPU-only session manager; **not integrated with the GPU runner yet**.

## Layout

`<prepared_run>/sessions/session-001/{session_manifest.json,checkpoints/}`

`<prepared_run>/latest_checkpoint.json` points to the newest SHA-256-verified checkpoint.

Each execution reserves a new `session-NNN` directory. Completed or interrupted session history is retained. A rolling checkpoint *within an active session* may be replaced atomically by Stage 3, but older sessions' finalized checkpoints must never be changed. The `run_manifest.json`, source snapshot, and resolved config are never mutated.

## Coordination contract (required in Stage 3)

1. Acquire `experiment_lock(run)` **before** calling `start_session`; hold it through training, checkpoint commits, and `finish_session`.
2. Validate checkpoint state with `t04_resume_state.load_checkpoint` (epoch, manifest identity, config digest, RNG etc.) before restoring any training state.
3. Write complete checkpoint atomically to the active session's `checkpoints/` using `atomic_torch_save`; then call `publish_checkpoint`.
4. Ensure session-local epoch CSV and checkpoint are transactionally reconciled before publish. The atomic latest pointer is the commit boundary. Stage 2 does not implement epoch CSV recovery.
5. On normal completion call `finish_session(..., status='completed')`; on caught exception use `status='interrupted'`. Abrupt SIGKILL may leave `status=running`, which is preserved for forensics; the OS releases the advisory lock.
6. Never change model architecture, optimizer setup, dataset split hashes, seed, or other training identity during an exact resume. Only increase target total epochs. Stage 3 must verify identity and reject changes.

**Scope limit:** This module provides *storage metadata and locking primitives*, not operational resume, robust interrupted writes, or a proven equivalence guarantee. It intentionally does not modify `t04_runner.py`, `moment_training.py`, or existing completed experiments.

On HPC, Slurm steps must use a filesystem supporting `flock` reliably and all processes must honor the same lock. Verify this on the actual shared filesystem.

Run: `python -m pytest -q tests/test_t04_sessions.py tests/test_t04_resume_state.py`
