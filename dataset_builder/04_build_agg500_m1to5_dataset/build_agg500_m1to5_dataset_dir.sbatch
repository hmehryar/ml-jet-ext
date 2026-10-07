#!/usr/bin/env bash
#SBATCH --job-name=agg500m15
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --time=12:00:00
##SBATCH --mem=16G
##SBATCH -q primary

#SBATCH --partition=ecscp
##SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=504:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=express


set -euo pipefail

echo "=================================================="
echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Run dir     : ${RUN_DIR:-UNSET}"
echo "Host        : $(hostname)"
echo "Start time  : $(date)"
echo "=================================================="

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: RUN_DIR is not set."
  exit 1
fi

PYTHON_SCRIPT="04_build_agg500_m1to5_dataset.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: ${PYTHON_SCRIPT} not found in $(pwd)"
  exit 1
fi

# Activate your environment here if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch

python "${PYTHON_SCRIPT}" \
  --run_dir "${RUN_DIR}" \
  --base_root "/rs/rs_grp_mljet/ml-jet-ext" \
  --unfolded_dir_name "pTHat50-250_eCM5020_c0005_pT90-110_R0p4" \
  --old_agg_dir_name "pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500" \
  --new_agg_dir_name "pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5_normalized" \
  --normalize_moments

echo "End time: $(date)"