# Isolated T03 development and training

`train_t03.ipynb` delegates to reusable Python modules. The canonical CLI is
`python tools/run_t03.py`. Historical notebooks exports, configs, and Slurm
launchers remain historical interfaces; use the snapshot launcher for new jobs.
No launcher submits other jobs automatically.

## Overrides and compatibility

YAML values remain defaults. Explicit flags override YAML before configuration
resolution: `--dataset-root`, `--checkpoint`, `--output-root`, `--workers`,
`--device`, and `--epochs`. The existing `get_config(path)` interface and legacy
training notebook retain their behavior. YAML dataset_root still takes precedence
over DATASET_ROOT unless an explicit CLI override is supplied.

The new runner intentionally uses fresh output paths:
`ROOT/{local,hpc}/EXPERIMENT/seed-SEED/UTC-UUID/`.
ROOT defaults to the YAML output_dir; `--output-root` changes it.
`--run-id` can name a run explicitly; an occupied directory is always rejected.
Preparation failures leave a reserved directory as evidence; choose a new ID.
There is no automatic resume or retry into an existing run.

## Scheduler

Selection and early stopping remain validation joint accuracy/max.
Scheduler monitor is independently explicit: `val_joint_accuracy/max` or
`val_loss/min`. A missing monitor defaults to val_joint_accuracy for compatibility.
Inconsistent historical YAMLs are rejected with an actionable error; they are
never silently rewritten. For old v3 configs, reproduce the saved job script's
accuracy scheduler with `--scheduler-monitor val_joint_accuracy --scheduler-mode max`.
To intentionally schedule on loss, specify both loss/min flags. Resolved settings,
including min_lr, are recorded; this is different from historical scripts that
ignored the YAML min_lr. Keep old outputs unchanged and identify new procedures.

## Local pilot

On a local development GPU (not an HPC login node):

Use a local SSD dataset copy with the same published split CSVs. Reading thousands
of small arrays through SSHFS can leave the GPU waiting on I/O; a mounted HPC
directory is useful for inspection but is a poor default for local full epochs.
Preserve the canonical data and metadata when staging a separate local copy.

```bash
python tools/run_t03.py \
  --config experiments/exp_03_reporting_pilot/config/E2_m1_m2_pretrained.yml \
  --dataset-root /local/path/to/normalized-dataset \
  --checkpoint /local/path/to/JetBench/best_model.pth \
  --profile local --output-root /local/path/to/runs --workers 0 --device cuda
```

`auto` retains T03's CUDA requirement; an explicit `--device cpu` is available
for development only. The GPU validation suite refuses CPU execution.

## Production staging and execution

Freeze source BEFORE submission, while no agent is editing that checkout:

```bash
python tools/run_t03.py \
  --config experiments/exp_03_cumulative_moment_ablation_v3/config/E2_m1_m2_pretrained.yml \
  --profile hpc --output-root /absolute/hpc/runs --workers 1 --device cuda \
  --scheduler-monitor val_joint_accuracy --scheduler-mode max --prepare-only
```

Preparation loads no accelerator libraries or tensors. It reads split metadata
and hashes the checkpoint, so stage it locally or on a compute node if that I/O
would be substantial on a login node. It does not evaluate held-out samples.

Only after validating resources and authorizing submission:

```bash
mkdir -p logs
export MLJET_PYTHON=/absolute/path/to/environment/bin/python
sbatch slurm/train_t03_snapshot.slurm /absolute/prepared/run
```

The provided partition/QOS/GRES are inherited examples, not verified site policy.
Override them through sbatch as appropriate. The new launcher has no hardcoded
account, repository, dataset, checkpoint, or interpreter path. Do not use historical
launchers for snapshot isolation. Unit tests run separately before GPU jobs.

Each run archives project Python modules, the notebook, its exported script,
launchers, tests, raw/resolved config, Git revision/status/diff, hashes, package
versions, hardware/runtime settings, imported module paths, and Slurm metadata.
Source is read-only, checked against hashes, and executed in a fresh process with
only the snapshot on PYTHONPATH. Checkpoints and split files are hashed before
and after training; changing them invalidates the run. Dataset arrays must remain
immutable under the published dataset contract; they are not copied or hashed
individually. Private agent/environment files are excluded from source snapshots.
Read-only modes prevent accidental edits, not privileged tampering.

## Validation gate

Run synthetic tests locally first, then prepare a distinct HPC validation run
with `--validation-suite --prepare-only`. It verifies checkpoint transfer and
adapter gradients for all five representations, then trains the selected moments
for one complete training epoch and full validation. It records the transfer
results and epoch completion. No held-out test evaluation occurs. Review that
run before authorizing another production batch. This is not an exact-resume test.

Reports use the best checkpoint and the validation loader only. Reporting failures
are recorded separately and do not discard successful training artifacts.
