# T04-10 Stage 8: Open-ended, append-only continuation

## Scope

Continue the 12 audited CLIP/SigLIP single-head experiments from their last
SHA-256-verified checkpoint, initially epoch 80. The per-submission **total**
epoch target can be **any positive integer greater than the last committed
epoch**. `200` is the initial requested milestone, **not a hard cap**. A new
Slurm job can resume to 201, 300, 1000, etc. without changing the original
80-epoch configuration or code again.

The original immutable `run/source`, `run_manifest.json`, `config.resolved.json`,
and prior sessions are preserved. A fresh append-only session is created with
`parent_checkpoint`, `target_epochs`, and `continuation_protocol` provenance.
The optimizer, scheduler, model, RNG state, global history, and best-validation
macro-F1 are restored from the latest published checkpoint (not the best-only
model checkpoint). Training continues at `last_epoch + 1`.

## Safety

- Extension source is frozen separately under
  `t04_10_single_head/_continuation_protocols/continuation-<commit>/source`.
- The protocol manifest hashes all Python files in that snapshot.
- Both original prepared-run files and the extension snapshot are verified before
  job execution and again under the experiment lock.
- Parent must belong to the exact frozen 12-run Stage 6 production matrix,
  whose `git_commit` is `c866804e38eef508e83297c111ab4318278eeb7e`.
- The existing 80-epoch history remains the initial segment of extended history;
  original checkpoints are never moved or replaced. Global `history.json`,
  metrics CSV and `latest_checkpoint.json` advance as new epochs commit.
- Retained checkpoints follow Stage 6 rolling/latest-and-best per-session rules;
  older session files are never changed.
- Slurm jobs start held and are only released after resource checks.
- No T04-07 paths or code are written by the continuation controller.
- The test split remains excluded from training/model selection. Source-overlap
  disclosure (103,221 occurrences) remains applicable.

## On HPC, after installing Stage 8 and committing it to the feature branch

```bash
cd ~/ml-jet-ext
conda activate pytorch
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m train_utils.t04_10_extension --prepare-protocol --repo "$PWD"

# Preview/read-only checks all 12 parents:
bash tools/t04_10_extend_all.sh 200

# Submits 12 jobs held; verifies 1 CPU, 1 A100 MIG, 12G, express/ecscp;
# then releases all after successful audit:
bash tools/t04_10_extend_all.sh 200 --apply
```

**Important:** Run the preview first. The `--apply` script is NOT designed to
be re-run after partial submission without investigating its held job IDs.

Later, after some or all jobs have completed, repeat with a new target:

```bash
bash tools/t04_10_extend_all.sh 300
bash tools/t04_10_extend_all.sh 300 --apply
```

The target must be greater than **each** run's latest committed epoch. If some
runs need a different target or some jobs timed out, launch those individually
using `sbatch --hold --export=... slurm/t04_10_continuation.slurm` with the same
frozen protocol and guard, rather than modifying the completed snapshots.

## Failure/restart semantics

Each Slurm allocation is limited to 12 hours, but **training itself is not
limited to 12 hours**: an interrupted or timed-out allocation leaves the last
atomically published checkpoint, and the next job starts at its next epoch.
Do not run concurrent continuation jobs for the same experiment. The advisory
lock ensures only one executes, but duplicate submissions waste allocations.

## Acceptance

CPU regressions include synthetic epoch-80 to epoch-81/82 continuation,
training-history continuity, checksum tamper rejection and original Stage 6
session tests. GPU acceptance remains to be checked on csc1: at least one
continuation must produce `session-002` (or `session-003` for the previously
resumed seed-42 CLIP alpha run), a SHA-256 verified epoch-81 checkpoint,
and an unbroken 81-row history before considering the whole rollout complete.
