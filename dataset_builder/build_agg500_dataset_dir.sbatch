#!/usr/bin/env bash
#SBATCH --job-name=agg500_dir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH -q primary

set -euo pipefail

echo "Running on host: $(hostname)"
echo "RUN_DIR (config): ${RUN_DIR:-UNSET}"

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: RUN_DIR not set. sbatch must be called with --export=RUN_DIR=<config_name>."
  exit 1
fi

# Activate your environment (adapt to your grid setup)
# Example:
# module load anaconda3
# source activate pytorch
# or:
# source ~/.bashrc
# conda activate pytorch

# Paths for this job (GRID paths)
INPUT_ROOT="/rs/rs_grp_mljet/ml-jet-ext/pTHat50-250_eCM5020_c0005_pT90-110_R0p4"
AGG_ROOT="/rs/rs_grp_mljet/ml-jet-ext/pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500"

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "AGG_ROOT:   ${AGG_ROOT}"

python 03_build_agg500_dataset.py \
  --input-root "${INPUT_ROOT}" \
  --agg-root "${AGG_ROOT}" \
  --config "${RUN_DIR}" \
  --n-agg 400 \
  --group-size 500 \
  --max-pt 110.0 \
  --epsilon 1e-8 \
  --rng-seed 42
