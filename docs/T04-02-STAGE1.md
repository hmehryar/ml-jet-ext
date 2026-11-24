# T04-02 Stage 1: resumable-state primitives

This patch **does not make the current T04 runner resumable yet**. It introduces tested infrastructure for *new* runs: atomic checkpoint serialization, complete RNG and optimizer/scheduler state capture, strict identity validation, deterministic per-epoch DataLoader reseeding, and CSV last-row reconciliation.

The T04-01 completed pilots and validation run remain untouched and are **not** valid exact-resume checkpoints, because their `last_model.pth` files lack RNG and early-stop state.

## Next stage (required before T04-02 closure)

1. Wire these primitives into `moment_training.py` using a new opt-in resumable mode, retaining the old legacy behavior.
2. Add atomic, durable end-of-epoch transactions for `last_model.pth`, metrics history, and best checkpoint; account for interruption between these writes. Preserve best checkpoint and early-stop state.
3. Add safe execution-lock acquisition/release that allows a subsequent resume, plus a separate run-completion marker.
4. Pin immutable source and dataset inputs; allow only an explicit **total-epoch target extension** in the runtime invocation, never an uncontrolled alteration of the frozen configuration.
5. Verify deterministic worker seeding, sampling order, mid-epoch restart semantics, fixed GPU type and precision; test with the actual end-to-end T04 S1 runner.
6. Define handling for already stopped runs and strictly prevent independent second-stage fine-tuning under the same run identity.

Run tests: `python -m pytest -q tests/test_t04_resume_state.py`.
