#!/bin/bash
#SBATCH --job-name=t04-gpu-smoke
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$HOME/ml-jet-ext"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
: "${T04_SMOKE_MODEL:?clip or siglip}"
: "${T04_SMOKE_STRATEGY:?S1 S2 S3 or S4}"
: "${T04_SMOKE_REVISION:?pinned checkpoint SHA}"
BATCH="${T04_SMOKE_BATCH_SIZE:-1}"
mkdir -p "${HOME}/t04-03-gpu-smoke-results"
OUTPUT="${HOME}/t04-03-gpu-smoke-results/${T04_SMOKE_MODEL}_${T04_SMOKE_STRATEGY}_b${BATCH}_${SLURM_JOB_ID}.json"
python -m tools.t04_gpu_smoke \
  --model "$T04_SMOKE_MODEL" --strategy "$T04_SMOKE_STRATEGY" \
  --revision "$T04_SMOKE_REVISION" --channels 5 --batch-size "$BATCH" \
  --checkpoint-roundtrip --output "$OUTPUT"
