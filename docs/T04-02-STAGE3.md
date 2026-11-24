# T04-02 Stage 3 — immutable GPU training sessions

This patch **adds** session-based execution. It does **not** modify the original
T04 runner or the existing T03 moment training loop. Existing run output is not
eligible for exact resume, because its checkpoint lacks RNG/early-stopping state.

## Scope

* Session checkpoint per **completed epoch** (`sessions/session-NNN/checkpoints/epoch_000001.pth`).
  This is storage-intensive, but avoids ever overwriting the checkpoint referred
  to by the atomic latest pointer. Retention optimization belongs to a later patch.
* `latest_checkpoint.json` changes atomically at each epoch, followed by durable
  `moment_epoch_metrics.csv` append. On resume, one missing final row is rebuilt
  from the committed checkpoint. A CSV ahead of the checkpoint is rejected.
* `session-001` fresh, `session-002` resume, preserving historical files.
* Advisory lock held by the parent process across the entire GPU subprocess.
* New `--target-epochs` is a **total number of epochs**. It can extend the
  original plan without changing the immutable configuration; the extension is
  logged in the session manifest.
* Early-stopped trajectories may not be resumed merely by raising the epoch
  target. That would be a different experiment.
* Deterministic epoch seeds use `train_loader.generator` and unchanged data loader
  worker configurations. Bitwise equivalence across different GPU/software
  environments is not guaranteed.
* Only S1 frozen, FP32, cross entropy, accumulation=1 is supported. S2–S4, AMP
  and new losses are separate follow-up issues.

## Usage

**After installing this patch, prepare a NEW experiment.** Existing prepared
snapshots are immutable and do not include these modules.

```bash
python -m train_utils.t04_preflight \
  --config experiments/t04/config/production/siglip_s1_e5.yml \
  --prepare-only --run-id siglip-s1-e5-resume-validation-NEW

RUN="$PWD/training_output/T04/production/siglip/S1/E5/seed-42/siglip-s1-e5-resume-validation-NEW"
python -m train_utils.t04_preflight --verify "$RUN"

# For 2-epoch fresh followed by extension to 4, initial config epochs must be >= 2;
# target 4 is total epochs, not 4 extra epochs.
sbatch --export=ALL,T04_PREPARED_RUN="$RUN",T04_SESSION_MODE=fresh,T04_TARGET_EPOCHS=2 slurm/train_t04_session.slurm
# After the first job COMPLETES:
sbatch --export=ALL,T04_PREPARED_RUN="$RUN",T04_SESSION_MODE=resume,T04_TARGET_EPOCHS=4 slurm/train_t04_session.slurm
```

Before GPU jobs, run the Stage 1/2 + Stage 3 tests and existing regressions.

**Safety:** Do not manually delete/change `.session-execution.lock` while a job
runs. The OS unlocks advisory locks upon exit. A `target_epochs` extension is
not the same as changing model/train settings.
