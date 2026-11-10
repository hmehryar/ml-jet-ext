#!/usr/bin/env bash
# Default: read-only preview. Only --apply invokes sbatch, always with --hold.
set -euo pipefail
ROOT='/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head'
LOGS="$ROOT/slurm_logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
usage() { echo 'Usage: bash tools/t04_10_submit_held.sh /absolute/prepared/run [fresh|resume] [--apply]' >&2; exit 2; }
(( $# >= 1 && $# <= 3 )) || usage
RUN="$1"
MODE='fresh'
APPLY=0
shift
for arg in "$@"; do
  case "$arg" in
    fresh|resume) MODE="$arg" ;;
    --apply) APPLY=1 ;;
    *) usage ;;
  esac
done
cd "$REPO"
# Validates both source snapshot and the dedicated T04-10 namespace.
PYTHONPATH="$REPO" python "$REPO/tools/t04_10_stage5a_guard.py" \
  --run "$RUN" --mode "$MODE"
if (( APPLY == 0 )); then
  printf 'PREVIEW_ONLY: would submit exactly one HELD T04-10 %s job. No directories or jobs created.\n' "$MODE"
  exit 0
fi
# Batch log directory must exist at submission; mkdir is restricted to T04-10.
mkdir -p -- "$LOGS"
RUN="$(realpath "$RUN")"
JOB=$(sbatch --parsable --hold \
  --export=ALL,T04_10_PREPARED_RUN="$RUN",T04_10_MODE="$MODE",T04_10_REPO="$REPO" \
  "$REPO/slurm/t04_10_single_head_held.slurm")
printf 'SUBMITTED_HELD job=%s mode=%s run=%s logs=%s\n' "$JOB" "$MODE" "$RUN" "$LOGS"
printf 'Inspect squeue -j %s before any explicit release. Do not release without a resource check.\n' "${JOB%%;*}"
