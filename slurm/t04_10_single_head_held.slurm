#!/usr/bin/env bash
#SBATCH --job-name=T04_10_single
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/T04_10_%j.out
#SBATCH --error=slurm_logs/T04_10_%j.err
set -euo pipefail
: "${T04_10_PREPARED_RUN:?Must export the immutable prepared-run directory}"
: "${T04_10_MODE:?Must export fresh or resume}"
[[ "$T04_10_MODE" == fresh || "$T04_10_MODE" == resume ]]
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
RUN="$(realpath "$T04_10_PREPARED_RUN")"
cd "$RUN/source"
export PYTHONPATH="$RUN/source"
python -m train_utils.t04_10_preflight --verify "$RUN"
exec python "$RUN/source/train_utils/t04_10_runner.py" --prepared-run "$RUN" --mode "$T04_10_MODE"
