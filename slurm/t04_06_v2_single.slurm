#!/usr/bin/env bash
#SBATCH --job-name=T04-V2-GATE
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=72:00:00
#SBATCH --output=/rs/rs_grp_mljet/ml-jet-ext/slurm_logs/T04/a100/%j.out
#SBATCH --error=/rs/rs_grp_mljet/ml-jet-ext/slurm_logs/T04/a100/%j.err
set -euo pipefail
: "${T04_06_RUN:?}"
: "${T04_06_MODE:?}"
: "${T04_06_TARGET:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export T04_EXECUTION_PROFILE=cpu1_workers0_v1 T04_DATALOADER_WORKERS=0
python -m tools.t04_04_pilot execute --run "$T04_06_RUN" --mode "$T04_06_MODE" --target-epochs "$T04_06_TARGET"
