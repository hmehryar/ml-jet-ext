#!/usr/bin/env bash
#SBATCH --job-name=T04-07-GPU-SMOKE
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
set -euo pipefail
: "${T04_07_SMOKE_PHASE:?Set fresh or resume explicitly}"
: "${T04_07_SMOKE_MANIFEST:?Set absolute smoke manifest path}"
case "$T04_07_SMOKE_PHASE" in
  fresh) MODE=fresh; TARGET=1;;
  resume) MODE=resume; TARGET=2;;
  *) echo "Invalid smoke phase" >&2; exit 2;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from the repo root}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
unset T04_EXECUTION_PROFILE T04_DATALOADER_WORKERS || true
python -m tools.t04_07_gpu_smoke --manifest "$T04_07_SMOKE_MANIFEST" probe
RUN="$(python -m tools.t04_07_gpu_smoke --manifest "$T04_07_SMOKE_MANIFEST" get-run --index "$SLURM_ARRAY_TASK_ID")"
echo "SMOKE_PHASE=$T04_07_SMOKE_PHASE INDEX=$SLURM_ARRAY_TASK_ID RUN=$RUN"
nvidia-smi
python -m tools.t04_04_pilot execute --run "$RUN" --mode "$MODE" --target-epochs "$TARGET"
