#!/usr/bin/env bash
#SBATCH --job-name=T04-06-E5
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail
: "${T04_06_ARRAY_MANIFEST:?Manifest path required}"
: "${SLURM_ARRAY_TASK_ID:?Array task required}"
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUN="$(python -m tools.t04_06_array_task --manifest "$T04_06_ARRAY_MANIFEST" --index "$SLURM_ARRAY_TASK_ID")"
echo "T04-06 E5 index=${SLURM_ARRAY_TASK_ID} run=${RUN}"
python -m tools.t04_04_pilot execute --run "$RUN" --mode fresh --target-epochs 80
