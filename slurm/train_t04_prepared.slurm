#!/bin/bash
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
set -euo pipefail
: "${T04_PREPARED_RUN:?Export T04_PREPARED_RUN as absolute path to prepared run}"
cd /wsu/home/gy/gy40/gy4065/ml-jet-ext
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python -m train_utils.t04_snapshot_execute --prepared-run "$T04_PREPARED_RUN"
