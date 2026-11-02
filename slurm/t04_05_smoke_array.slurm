#!/usr/bin/env bash
#SBATCH --job-name=T04-05-E5
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
set -euo pipefail
: "${T04_05_MANIFEST:?Absolute path to immutable smoke manifest required}"
: "${T04_05_MODE:?fresh or resume required}"
: "${T04_05_TARGET:?Target epoch required}"
[[ "$T04_05_MODE" == fresh && "$T04_05_TARGET" == 1 ]] || \
[[ "$T04_05_MODE" == resume && "$T04_05_TARGET" == 2 ]] || { echo 'Invalid mode/target phase' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Run from repository root}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
RUN="$(python -m tools.t04_05_smoke get-run --manifest "$T04_05_MANIFEST" --index "$SLURM_ARRAY_TASK_ID")"
echo "T04-05 array index=$SLURM_ARRAY_TASK_ID mode=$T04_05_MODE target=$T04_05_TARGET run=$RUN"
nvidia-smi
python -m tools.t04_04_pilot execute --run "$RUN" --mode "$T04_05_MODE" --target-epochs "$T04_05_TARGET"
