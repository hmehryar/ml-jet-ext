#!/usr/bin/env bash
#SBATCH --job-name=T03v2
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO_ROOT="/wsu/home/gy/gy40/gy4065/ml-jet-ext"
PYTHON="/wsu/home/gy/gy40/gy4065/.conda/envs/pytorch/bin/python"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_ROOT="/rs/rs_grp_mljet/ml-jet-ext/pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5_normalized"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

CONFIG_PATH="${1:?Usage: sbatch slurm/train_t03_moments_v2.slurm CONFIG.yml}"
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: missing YAML: $CONFIG_PATH" >&2; exit 1; }
[[ -f train_t03.ipynb ]] || { echo 'ERROR: train_t03.ipynb missing' >&2; exit 1; }
[[ -f tools/notebook_to_py.py ]] || { echo 'ERROR: tools/notebook_to_py.py missing' >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "ERROR: Python not executable: $PYTHON" >&2; exit 1; }

EXPERIMENT_NAME="$(basename "$CONFIG_PATH" .yml)"
JOB_ID="${SLURM_JOB_ID:?This script is intended for sbatch}"
SCRIPT_PATH="$REPO_ROOT/train_t03_${EXPERIMENT_NAME}_${JOB_ID}.py"

echo "============================================================"
echo "Experiment : $EXPERIMENT_NAME"
echo "Job name   : ${SLURM_JOB_NAME:-unknown}"
echo "Job ID     : $JOB_ID"
echo "Node       : $(hostname)"
echo "YAML       : $CONFIG_PATH"
echo "Script     : $SCRIPT_PATH"
echo "Time limit : 24:00:00"
echo "Memory     : 12G"
echo "============================================================"

# Check inputs before creating/running the training script.
"$PYTHON" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text())

# Validate cumulative moment channels
moments = cfg.get("moments")
assert moments in [
    list(range(1, n + 1)) for n in range(1, 6)
], f"Invalid moments: {moments}"

# Let the YAML control the number of epochs
epochs = int(cfg.get("epochs", 0))
assert epochs > 0, f"Invalid epochs: {epochs}"

# Validate Gaussian initialization when configured
initialization = cfg.get("initialization")

if isinstance(initialization, dict):
    init_type = initialization.get("type")

    if init_type == "gaussian":
        assert float(initialization.get("std", 0)) > 0, (
            f"Invalid Gaussian initialization: {initialization}"
        )

# Verify pretrained checkpoint exists
checkpoint_path = cfg.get("checkpoint_path")

assert checkpoint_path, "Missing checkpoint_path"

assert Path(checkpoint_path).is_file(), (
    f"Checkpoint not found: {checkpoint_path}"
)

print("========================================")
print("Configuration validated")
print("Experiment:", config_path.stem)
print("Moments:", moments)
print("Epochs:", epochs)
print("Initialization:", initialization)
print("Checkpoint:", checkpoint_path)
print("========================================")
PY

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

# Validate reporting before spending GPU time on training.
"$PYTHON" -m pytest tests/test_t03_reporting.py -q

# Convert notebook with the lightweight converter: no Jupyter dependency.
# Persist the exact generated Python source in the repository root.
"$PYTHON" tools/notebook_to_py.py train_t03.ipynb "$SCRIPT_PATH"
"$PYTHON" -m py_compile "$SCRIPT_PATH"

# Python script is stored permanently; do not delete it on job exit.
echo "Starting experiment: $EXPERIMENT_NAME"
"$PYTHON" -u "$SCRIPT_PATH" --config "$CONFIG_PATH"
echo "Completed experiment: $EXPERIMENT_NAME"
