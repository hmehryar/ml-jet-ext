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

cd /wsu/home/gy/gy40/gy4065/ml-jet-ext

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Configuration: ${T04_CONFIG}"

nvidia-smi

python -m train_utils.t04_runner \
    --config "${T04_CONFIG}"
