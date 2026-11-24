#!/bin/bash
#SBATCH --job-name=t04-session
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
set -euo pipefail
cd /wsu/home/gy/gy40/gy4065/ml-jet-ext
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
: "${T04_PREPARED_RUN:?Set T04_PREPARED_RUN}"
: "${T04_SESSION_MODE:?Set T04_SESSION_MODE to fresh or resume}"
: "${T04_TARGET_EPOCHS:?Set T04_TARGET_EPOCHS}"
python -m train_utils.t04_session_execute \
  --prepared-run "$T04_PREPARED_RUN" \
  --mode "$T04_SESSION_MODE" \
  --target-epochs "$T04_TARGET_EPOCHS"
