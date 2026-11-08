#!/usr/bin/env bash
#SBATCH --job-name=T04-07-PROD
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=72:00:00
set -euo pipefail
: "${T04_07_PROD_RUN:?Explicit prepared run required}"
: "${T04_07_PROD_MODE:?Explicit training mode required}"
: "${T04_07_PROD_TARGET:?Explicit epoch target required}"
[[ "$T04_07_PROD_MODE" == fresh || "$T04_07_PROD_MODE" == resume ]] || exit 2
[[ "$T04_07_PROD_TARGET" == 80 ]] || { echo 'This production script targets epoch 80 only' >&2; exit 2; }
[[ "$T04_07_PROD_RUN" == /rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_07_loss_pilot/* || "$T04_07_PROD_RUN" == /.rs/volumes/rs_grp_mljet/*/ml-jet-ext/training_output/T04/t04_07_loss_pilot/* ]] || { echo 'Run outside T04-07 production namespace' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from checkout root}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export T04_EXECUTION_PROFILE=cpu1_workers0_v1 T04_DATALOADER_WORKERS=0
python -m train_utils.t04_preflight --verify "$T04_07_PROD_RUN"
python -m tools.t04_04_pilot execute --run "$T04_07_PROD_RUN" --mode "$T04_07_PROD_MODE" --target-epochs "$T04_07_PROD_TARGET"
