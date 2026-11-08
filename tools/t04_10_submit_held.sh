#!/usr/bin/env bash
# Explicitly creates a HELD job, never starts training automatically.
set -euo pipefail
RUN="${1:?Usage: bash tools/t04_10_submit_held.sh /absolute/prepared/run [fresh|resume]}"
MODE="${2:-fresh}"
[[ "$MODE" == fresh || "$MODE" == resume ]] || { echo 'Invalid mode' >&2; exit 2; }
[[ -f "$RUN/run_manifest.json" ]] || { echo 'Not a prepared run' >&2; exit 2; }
mkdir -p slurm_logs
sbatch --hold --export=ALL,T04_10_PREPARED_RUN="$(realpath "$RUN")",T04_10_MODE="$MODE" slurm/t04_10_single_head_held.slurm
printf 'Job submitted HELD. Inspect squeue and use scontrol release JOB_ID only after approval.\n'
