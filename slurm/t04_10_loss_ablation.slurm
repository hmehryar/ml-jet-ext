#!/usr/bin/env bash
#SBATCH --job-name=T04_10_loss
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/slurm_logs/T04_10_LOSS_%j.out
#SBATCH --error=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/slurm_logs/T04_10_LOSS_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
: "${T04_10_LOSS_RUN:?}" "${T04_10_LOSS_TARGET:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
RUN="$(realpath "$T04_10_LOSS_RUN")"
[[ -f "$RUN/source/train_utils/t04_10_loss_runner.py" ]] || { echo 'Missing frozen loss runner' >&2; exit 2; }
export PYTHONPATH="$RUN/source"
cd "$RUN/source"
python "$RUN/source/train_utils/t04_10_loss_runner.py"   --run "$RUN" --target-epochs "$T04_10_LOSS_TARGET" --execute
