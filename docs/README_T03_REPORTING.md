# T03 notebook-first experiment reporting

- `train_t03.ipynb` is the **only maintained training entry point**. Do not hand-edit `train_t03.py`.
- Slurm converts a **job-private copy** of the notebook into a Python script in `$SLURM_TMPDIR` or `/tmp`, then executes it with `--config`. No concurrent overwrite of repository files.
- Pass a distinct job name with `sbatch --job-name=T03_E2_m1_m2_pretrained ...`; Slurm logs go to `logs/%x_%j.out` and `.err`.
- The reporting module loads the **best checkpoint**, evaluates **validation only**, and writes both PNG and PDF versions of curves, confusion matrices, and class/probability histograms.
- `train_utils/moment_training.py` contains the **minimal T02-compatible CSV extension** for `train_alpha_loss`, `val_alpha_loss`, `train_q0_loss`, and `val_q0_loss`. Check `git diff` before replacing your working local file; if it differs, apply only these added CSV fields instead.
- Existing one-epoch pilot CSVs without per-head loss columns remain readable: the reporter skips the missing loss curves.
- Training remains successful if reporting fails. Re-run plotting separately using the same model/dataloader initialization and `generate_t03_report` to avoid rerunning 50 epochs.

## Install

```bash
unzip -l ML_JET_Ext_T03_Reporting_Notebook_First.zip
unzip -o ML_JET_Ext_T03_Reporting_Notebook_First.zip -d .
python -m pytest tests/test_t03_reporting.py -v
mkdir -p logs
```

## Local pilot verification

Run `train_t03.ipynb` interactively or convert once for local testing:

```bash
jupyter nbconvert --to python train_t03.ipynb --output train_t03
PYTHONPATH="$PWD" python train_t03.py --config path/to/NEW/pilot.yml
```

Avoid rerunning the already-completed pilot using the same output path; the training entry point refuses overwrite.

## Slurm submission

```bash
EXP=experiments/exp_03_cumulative_moment_ablation/config
for i in 1 2 3 4 5; do
  CONFIG=$(ls "$EXP/E${i}_"*_pretrained.yml)
  NAME=$(basename "$CONFIG" .yml)
  sbatch --job-name="T03_${NAME}" slurm/train_t03_moments.slurm "$CONFIG"
done
```

Run from repository root. Verify GPUs, partition, access, and the 50-epoch configs first.
