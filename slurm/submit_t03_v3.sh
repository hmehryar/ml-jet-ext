#!/usr/bin/env bash
# Submit T03 v3 jobs with distinct names and log files.
set -euo pipefail

REPO_ROOT="/wsu/home/gy/gy40/gy4065/ml-jet-ext"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_DIR="experiments/exp_03_cumulative_moment_ablation_v3/config"
SLURM_SCRIPT="slurm/train_t03_moments_v3.slurm"
[[ -f "$SLURM_SCRIPT" ]] || { echo "ERROR: Missing $SLURM_SCRIPT" >&2; exit 1; }

if [[ $# -eq 0 ]]; then
    set -- E5
elif [[ $# -eq 1 && "$1" == "--all" ]]; then
    set -- E1 E2 E3 E4 E5
fi

# Validate every argument and configuration before submitting anything.
for ID in "$@"; do
    case "$ID" in
        E1) NAME="E1_m1_pretrained" ;;
        E2) NAME="E2_m1_m2_pretrained" ;;
        E3) NAME="E3_m1_m3_pretrained" ;;
        E4) NAME="E4_m1_m4_pretrained" ;;
        E5) NAME="E5_m1_m5_pretrained" ;;
        *) echo "ERROR: Invalid experiment '$ID' (use E1-E5 or --all)" >&2; exit 2 ;;
    esac
    CONFIG="$CONFIG_DIR/$NAME.yml"
    [[ -f "$CONFIG" ]] || { echo "ERROR: Missing $CONFIG" >&2; exit 1; }
done

for ID in "$@"; do
    case "$ID" in
        E1) NAME="E1_m1_pretrained" ;;
        E2) NAME="E2_m1_m2_pretrained" ;;
        E3) NAME="E3_m1_m3_pretrained" ;;
        E4) NAME="E4_m1_m4_pretrained" ;;
        E5) NAME="E5_m1_m5_pretrained" ;;
    esac
    CONFIG="$CONFIG_DIR/$NAME.yml"
    echo "Submitting T03v3_${NAME}: $CONFIG"
    sbatch \
        --job-name="T03v3_${NAME}" \
        --output="logs/T03v3_${NAME}_%j.out" \
        --error="logs/T03v3_${NAME}_%j.err" \
        "$SLURM_SCRIPT" "$CONFIG"
done
