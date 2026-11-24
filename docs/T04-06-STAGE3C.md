# T04-06 Stage 3C — Recovery eligibility (read-only)

Stage 3C is intentionally conservative. Run `python -m tools.t04_06_recovery inspect` and `python -m pytest -q tests/test_t04_06_recovery.py`.

This does not submit, resubmit, release, cancel, prepare or mutate runs; does not alter the planning or operational registry. **Do not execute registry transition commands based solely on this report.**

Policies:
- Planned and active entries cannot resume.
- Scheduler terminal evidence is mandatory. The standalone CLI supplies none, so it cannot authorize an action for a live job; the later Stage 3D handshake must bind the observation to the recorded job ID.
- Source, config SHA, pointer hash, checkpoint payload identity, optimizer/scheduler/RNG state, CSV epoch sequence and session state must agree.
- An unfinished `reserved` or `running` session is a hard block until inspected and explicitly reconciled; even if the kernel lock was released on job death. Never delete or rewrite that session.
- Early-stopped checkpoints cannot resume because `train_session` rejects them. Completing configured horizon is also terminal.
- A valid partial checkpoint with a **closed** audited session after a failed scheduler task may be reviewed for resume; `review_resume` is not automatic authorization.
- Missing final CSV row or incomplete runtime telemetry needs manual audit; never declare training complete on Slurm exit status alone.
- Don't change config `num_workers` or other hashed fields in-place; use an explicitly versioned new experiment instead.
- For Stage 3D, implement Slurm task ownership, registry compare-and-swap, verified parent checksum at transition, submission-intent fencing, and a recovery/resume integration test. Do not use this file as a standalone job submitter.

**One CPU:** `slurm/t04_06_array.slurm` has `--cpus-per-task=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. This patch deliberately does not modify it or the hash-bound 120 YAML configs.

**Known limitation:** The existing training runner may leave a reserved/running session after an interrupted process. This stage refuses automatic recovery of such an experiment; resolving it requires a separate audited session-finalization mechanism in Stage 3D, not manual deletion.
