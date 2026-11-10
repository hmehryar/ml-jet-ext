#!/usr/bin/env bash
# Read-only by default. --apply submits the audited 12 runs held, audits resources,
# then releases as one batch. Re-run only after diagnosing any partial submission.
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT='/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head'
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() { echo "Usage: bash tools/t04_10_extend_all.sh POSITIVE_TOTAL_TARGET_EPOCHS [--apply]" >&2; exit 2; }
(( $# == 1 || $# == 2 )) || usage
TARGET="$1"
[[ "$TARGET" =~ ^[1-9][0-9]*$ ]] || usage
[[ $# == 1 || "$2" == '--apply' ]] || usage
APPLY=0
(( $# == 1 )) || APPLY=1
cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'STOP: tracked tree dirty' >&2; exit 1; }
[[ "$(git branch --show-current)" == 'feat/t04-10-single-head-pilot' ]] || { echo 'STOP: wrong branch' >&2; exit 1; }
COMMIT="$(git rev-parse HEAD)"
PROTO="$ROOT/_continuation_protocols/continuation-${COMMIT:0:12}"
[[ -f "$PROTO/protocol_manifest.json" ]] || { echo "STOP: protocol not prepared: $PROTO" >&2; exit 1; }
RUNS=()
LABELS=()
for model in clip siglip; do
  for task in alpha_only q0_only; do
    if [[ "$model" == clip && "$task" == alpha_only ]]; then moment=E3
    elif [[ "$model" == siglip && "$task" == alpha_only ]]; then moment=E2
    else moment=E4; fi
    for seed in 42 123 2026; do
      run="$ROOT/$model/$task/$moment/seed-$seed/t04_10_${model}_${task}_seed${seed}_session_v1"
      RUNS+=("$run")
      LABELS+=("$model $task $seed")
    done
  done
done
[[ ${#RUNS[@]} -eq 12 ]] || exit 1
printf '\n=== Preflight all 12 immutable baseline runs, their last checkpoint, and frozen protocol ===\n'
for run in "${RUNS[@]}"; do
  PYTHONPATH="$PROTO/source" python "$PROTO/source/train_utils/t04_10_extension.py" \
    --run "$run" --protocol "$PROTO" --target-epochs "$TARGET"
done
if (( APPLY == 0 )); then
  echo 'ALL_12_PREVIEW_PASS; no jobs submitted'
  exit 0
fi
# Fail closed on a previous incomplete rollout; no duplicate batch submission.
if [[ -n "$(squeue -h -u "$USER" -n T04_10_continue -o '%i')" ]]; then
  echo 'STOP: T04_10_continue jobs are already active or pending' >&2
  exit 1
fi
mkdir -p "$ROOT/slurm_logs"
IDS=()
printf '\n=== Submit exactly 12 HELD continuation jobs ===\n'
for i in "${!RUNS[@]}"; do
  job="$(sbatch --parsable --hold \
    --export=ALL,T04_10_RUN="${RUNS[i]}",T04_10_PROTOCOL="$PROTO",T04_10_TARGET="$TARGET" \
    "$REPO/slurm/t04_10_continuation.slurm")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "STOP: bad job ID: $job" >&2; exit 1; }
  IDS+=("$job")
  printf 'HELD job=%s run=%s\n' "$job" "${LABELS[i]}"
done
printf '\n=== Audit resources before releasing any job ===\n'
for job in "${IDS[@]}"; do
  data="$(scontrol show job -o "$job")"
  for pattern in \
    '(^| )JobName=T04_10_continue( |$)' \
    '(^| )JobState=PENDING( |$)' \
    '(^| )Reason=JobHeldUser( |$)' \
    '(^| )NumCPUs=1( |$)' \
    '(^| )Partition=ecscp( |$)' \
    '(^| )QOS=express( |$)' \
    '(^| )TimeLimit=12:00:00( |$)' \
    '(^| )ReqTRES=[^ ]*cpu=1,mem=12G,[^ ]*gres/gpu=1,[^ ]*gres/gpu:nvidia_a100_80gb_pcie_1g\.10gb=1( |$)' \
    '(^| )TresPerNode=gres:gpu:nvidia_a100_80gb_pcie_1g\.10gb:1( |$)'; do
    grep -Eq "$pattern" <<< "$data" || { echo "STOP: invalid resources for $job" >&2; exit 1; }
  done
  echo "RESOURCE_PASS job=$job"
done
printf '\n=== Release 12 audited jobs ===\n'
for job in "${IDS[@]}"; do
  data="$(scontrol show job -o "$job")"
  grep -Eq '(^| )Reason=JobHeldUser( |$)' <<< "$data" || { echo "STOP: job $job no longer held" >&2; exit 1; }
  scontrol release "$job"
  echo "RELEASED job=$job"
done
printf '\nTARGET=%s\nCONTINUATION_12_RELEASED\n' "$TARGET"
joblist="$(IFS=,; echo "${IDS[*]}")"
squeue -j "$joblist" -o '%.12i %.22j %.10T %.12M %.28R' || true
