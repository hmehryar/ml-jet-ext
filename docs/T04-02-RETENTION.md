# T04-02: checkpoint retention in session-managed runs

`T04_CHECKPOINT_RETENTION=last_best` (default) keeps only the latest epoch recovery checkpoint and the **session-local best** checkpoint (if distinct). Each epoch is saved to a uniquely named file, the best checkpoint is hard-linked before publication, then the SHA-256-verified latest pointer is atomically published. Only after that commit are older files pruned **within the active session**. Completed sessions are never pruned. A failed cleanup cannot invalidate the latest pointer.

`T04_CHECKPOINT_RETENTION=all` disables pruning and retains every epoch checkpoint in each session for debugging and comparison. Set through the Slurm export environment, for example:

```bash
sbatch --export=ALL,T04_PREPARED_RUN="$RUN",T04_SESSION_MODE=fresh,T04_TARGET_EPOCHS=2,T04_CHECKPOINT_RETENTION=last_best slurm/train_t04_session.slurm
```

Best-selection uses validation joint accuracy and may point to a historical session. The prior session's preserved `best_epoch_*.pth` remains accessible. Checkpoints already produced before installing this change are not deleted or migrated automatically.

**Integrity caveat**: session-local best hard links are preserved inside completed sessions, but are not part of the `latest_checkpoint.json` verification contract. The latest recovery checkpoint remains SHA-256 checked. Best-checkpoint registration and verification may be expanded in Stage 4. The best checkpoint can contain the same epoch as the latest recovery checkpoint (two filenames but hard-linked storage).

This patch changes only the Stage 3 session-managed training path, not legacy T03/T04 training.
