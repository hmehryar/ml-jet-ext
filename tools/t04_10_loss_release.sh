#!/usr/bin/env bash
# Dry run by default. Only releases exactly the specified held T04-10 loss jobs.
set -euo pipefail
if (( $# < 1 )); then echo "Usage: bash tools/t04_10_loss_release.sh JOBID [JOBID ...] [--apply]" >&2; exit 2; fi
apply=0
if [[ "${@: -1}" == '--apply' ]]; then apply=1; set -- "${@:1:$#-1}"; fi
(( $# >= 1 )) || { echo 'No job IDs' >&2; exit 2; }
ids=("$@")
for id in "${ids[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid job id $id" >&2; exit 2; }
  data="$(scontrol show job -o "$id")"
  for pattern in \
    '(^| )JobName=T04_10_loss( |$)' \
    '(^| )JobState=PENDING( |$)' \
    '(^| )Reason=JobHeldUser( |$)' \
    '(^| )Partition=ecscp( |$)' \
    '(^| )QOS=express( |$)' \
    '(^| )NumCPUs=1( |$)' \
    '(^| )TimeLimit=12:00:00( |$)' \
    '(^| )ReqTRES=[^ ]*cpu=1,mem=12G,[^ ]*gres/gpu=1,[^ ]*gres/gpu:nvidia_a100_80gb_pcie_1g\.10gb=1( |$)' \
    '(^| )TresPerNode=gres:gpu:nvidia_a100_80gb_pcie_1g\.10gb:1( |$)'; do
      grep -Eq "$pattern" <<< "$data" || { echo "RESOURCE_FAIL $id pattern $pattern" >&2; exit 1; }
  done
  echo "RESOURCE_PASS $id"
done
if (( apply == 0 )); then echo 'PREVIEW_ONLY: no jobs released'; exit 0; fi
for id in "${ids[@]}"; do
  # Immediately recheck the job's identity + held state before release.
  data="$(scontrol show job -o "$id")"
  grep -Eq '(^| )JobName=T04_10_loss( |$)' <<< "$data" || { echo "REFUSED $id wrong name" >&2; exit 1; }
  grep -Eq '(^| )Reason=JobHeldUser( |$)' <<< "$data" || { echo "REFUSED $id not held" >&2; exit 1; }
  scontrol release "$id"
  echo "RELEASED $id"
done
