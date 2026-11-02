#!/bin/bash
#SBATCH --job-name=T04-04-GPU-Accept
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
cd "$HOME/ml-jet-ext"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
mkdir -p training_output/T04/gpu_acceptance
python -m tools.t04_04_gpu_acceptance \
  --output "training_output/T04/gpu_acceptance/t04_04_${SLURM_JOB_ID}.json"
