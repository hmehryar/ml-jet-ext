#!/bin/bash
#SBATCH --job-name=T04-05-MIG
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd /wsu/home/gy/gy40/gy4065/ml-jet-ext
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
: "${T04_FEAS_CONFIG:?Absolute configuration path required}"
: "${T04_FEAS_BATCH:?Batch size required}"
: "${T04_FEAS_REPORT:?Unique output path required}"
nvidia-smi
python -m tools.t04_05_feasibility --config "$T04_FEAS_CONFIG" --batch-size "$T04_FEAS_BATCH" --steps "${T04_FEAS_STEPS:-8}" --output "$T04_FEAS_REPORT"
