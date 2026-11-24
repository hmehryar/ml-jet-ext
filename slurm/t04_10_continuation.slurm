#!/usr/bin/env bash
#SBATCH --job-name=T04_10_continue
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/slurm_logs/T04_10_CONT_%j.out
#SBATCH --error=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/slurm_logs/T04_10_CONT_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
: "${T04_10_RUN:?}"
: "${T04_10_PROTOCOL:?}"
: "${T04_10_TARGET:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
printf 'T04_10_CONT_START job=%s host=%s cpus=%s target=%s\n' "${SLURM_JOB_ID:-?}" "$(hostname)" "${SLURM_CPUS_PER_TASK:-?}" "$T04_10_TARGET"
nvidia-smi -L
export PYTHONPATH="$T04_10_PROTOCOL/source"
cd "$T04_10_PROTOCOL/source"
exec python "$T04_10_PROTOCOL/source/train_utils/t04_10_extension.py" \
 --protocol "$T04_10_PROTOCOL" --run "$T04_10_RUN" \
 --target-epochs "$T04_10_TARGET" --execute
