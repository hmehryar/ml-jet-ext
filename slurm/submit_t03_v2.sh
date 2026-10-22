#!/usr/bin/env bash
# Submit each T03 v2 experiment as a separately named Slurm job.
set -euo pipefail
REPO_ROOT="/wsu/home/gy/gy40/gy4065/ml-jet-ext"
cd "$REPO_ROOT"
mkdir -p logs
CONFIG_DIR="experiments/exp_03_cumulative_moment_ablation_v2/config"
NAMES=(
  E1_m1_pretrained
  E2_m1_m2_pretrained
  E3_m1_m3_pretrained
  E4_m1_m4_pretrained
  E5_m1_m5_pretrained
)
# Validate all inputs before submitting any job.
for NAME in "${NAMES[@]}"; do
  CONFIG="$CONFIG_DIR/$NAME.yml"
  [[ -f "$CONFIG" ]] || { echo "ERROR: missing $CONFIG" >&2; exit 1; }
done
for NAME in "${NAMES[@]}"; do
  CONFIG="$CONFIG_DIR/$NAME.yml"
  echo "Submitting T03v2_$NAME ($CONFIG)"
  sbatch \
    --job-name="T03v2_$NAME" \
    --output="logs/T03v2_${NAME}_%j.out" \
    --error="logs/T03v2_${NAME}_%j.err" \
    slurm/train_t03_moments_v2.slurm "$CONFIG"
done
