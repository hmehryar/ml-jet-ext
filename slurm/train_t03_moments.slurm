#!/bin/bash
#SBATCH --job-name=T03_moments
#SBATCH --partition=ecscp
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=express
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /wsu/home/gy/gy40/gy4065/ml-jet-ext
CONFIG_PATH="${1:?Usage: sbatch --job-name=T03_E1_m1_pretrained slurm/train_t03_moments.slurm path/to/config.yml}"
test -f "$CONFIG_PATH"
PYTHON_BIN=/wsu/home/gy/gy40/gy4065/.conda/envs/pytorch/bin/python
test -x "$PYTHON_BIN"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg
JOB_DIR="${SLURM_TMPDIR:-/tmp}/mljet_t03_${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"
trap 'rm -rf "$JOB_DIR"' EXIT

echo "Experiment: $(basename "$CONFIG_PATH" .yml)"
echo "Slurm job: ${SLURM_JOB_ID}; Node: $(hostname)"
echo "Notebook: train_t03.ipynb"
echo "Config: $CONFIG_PATH"

# Generate a unique script for this Slurm job, avoiding concurrent writes to train_t03.py.
"$PYTHON_BIN" -m jupyter nbconvert --to python train_t03.ipynb \
  --output train_t03_run --output-dir "$JOB_DIR"
"$PYTHON_BIN" - <<'PYGPU'
import torch
assert torch.cuda.is_available(), 'CUDA GPU was not allocated'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PYGPU
"$PYTHON_BIN" -u "$JOB_DIR/train_t03_run.py" --config "$CONFIG_PATH"
