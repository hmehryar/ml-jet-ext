# CE plus ordinal distance for moment classification

The default remains cross-entropy. Opt in with a new experiment YAML:

```yaml
loss:
  type: ce_ordinal
  ordinal_weight: 1.0
  weights:
    alpha_output: 1.0
    q0_output: 1.0
```

For each head, the objective is `CE + ordinal_weight * ordinal`, where ordinal is
the batch mean of `sum_j w_j * (CDF_prediction(j) - CDF_target(j))**2` over the
first K-1 boundaries. The boundary weights are adjacent physical parameter gaps
divided by the full parameter range. They sum to one, giving each head a bounded,
dimensionless ordinal penalty. The total objective retains the existing head
weights. There is no regression head or second normalization of dataset tensors.

This is a weighted ranked probability / squared-CDF score, often called
squared-EMD in ordinal classification; it is not squared Wasserstein-2.
Reference: https://www3.cs.stonybrook.edu/~cvl/emd2.html

The existing alpha/Q0 logits, class mappings, argmax predictions, accuracy,
macro-F1, and joint accuracy are preserved. Checkpoint selection and early
stopping still maximize validation joint accuracy. The held-out test split is
never used for tuning, selection, or reports.

New CSV columns record CE and the ordinal component before lambda weighting,
plus per-head physical MAE, RMSE, and MAE normalized by the parameter range.
Physical errors use argmax class values: alpha is dimensionless; Q0 is in GeV.
Reports add loss-component and physical-MAE plots and best-checkpoint physical
metrics. Older CSVs/checkpoints remain readable; missing plots are skipped.
Checkpoints and run manifests record the loss settings. Model state keys are
unchanged. The loss coefficient is an experimental hyperparameter, not a validated
optimum; two epochs establish operational correctness, not superiority over CE.

## Two-epoch local pilot

Activate the private workstation environment, then run from the repository root:

```bash
source .local/env.local.sh
python -u tools/run_t03.py \
  --config experiments/exp_04_ordinal_loss/config/E2_m1_m2_ce_ordinal_pilot.yml \
  --checkpoint /absolute/accessible/JetBench/best_model.pth \
  --profile local --device cuda --workers 2 --epochs 2
```

Do not add `--validation-suite`: that separate workflow intentionally forces one
epoch. This pilot uses the fixed M1–M2 inputs and the same JetBench backbone
transfer, adapter initialization, optimizer, seed, and published primary splits.
A fresh run directory archives the resolved loss settings, raw YAML, source and
exported notebook, hashes, and runtime environment. No historical YAML is edited.

## HPC follow-up (instructions only; not automatic submission)

After reviewing the local results, use the same new pilot config for a two-epoch
HPC smoke test before increasing the training budget. On HPC, select the HPC
dataset with private environment settings or `--dataset-root`; clear a workstation
`MLJET_OUTPUT_ROOT` and select a fresh HPC output root explicitly if needed.
Verify current partition/QOS/GRES access and the selected Python environment.

Prepare on appropriate resources (checkpoint and split hashing can involve
substantial I/O; no training on login nodes):

```bash
unset MLJET_OUTPUT_ROOT MLJET_DATASET_ROOT
python tools/run_t03.py \
  --config experiments/exp_04_ordinal_loss/config/E2_m1_m2_ce_ordinal_pilot.yml \
  --profile hpc --device cuda --workers 1 --epochs 2 --prepare-only
```

The config retains original HPC dataset/checkpoint paths. If workstation
`MLJET_DATASET_ROOT` is inherited, unset it or explicitly override it first.
Preparation prints the immutable run directory. Only when submission is
authorized, submit that directory using the existing snapshot launcher:

```bash
mkdir -p logs
export MLJET_PYTHON=/absolute/path/to/environment/bin/python
sbatch slurm/train_t03_snapshot.slurm /absolute/prepared/run
```

Review `run_manifest.json` for both `status: complete` and `reporting: complete`,
two epoch CSV rows, consistent decomposition of per-head loss, physical error
metrics, and preserved split hashes. Compare with CE using matched budgets and
seeds on validation; retain the held-out test set for the frozen final procedure.
