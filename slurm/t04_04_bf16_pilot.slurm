#!/bin/bash
#SBATCH --job-name=T04-04-BF16
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
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
: "${T04_PILOT_RUN:?Set T04_PILOT_RUN to prepared run absolute path}"
: "${T04_PILOT_MODE:?Set T04_PILOT_MODE=fresh or resume}"
: "${T04_PILOT_TARGET:?Set T04_PILOT_TARGET=1 or 2}"
nvidia-smi
python -m tools.t04_04_pilot execute --run "$T04_PILOT_RUN" --mode "$T04_PILOT_MODE" --target-epochs "$T04_PILOT_TARGET"
