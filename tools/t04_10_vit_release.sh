#!/usr/bin/env bash
# Inspect one held T04_10_vitA job, fail closed on unexpected Slurm allocation.
set -euo pipefail
[[ $# == 1 || ( $# == 2 && $2 == --apply ) ]] || { echo 'Usage: t04_10_vit_release.sh JOB_ID [--apply]' >&2; exit 2; }
JOB="$1"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Numeric job ID required' >&2; exit 2; }
DETAIL="$(scontrol show job "$JOB" -o)"
for REQUIRED in 'JobName=T04_10_vitA' 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=ecscp' 'QOS=express' 'NumCPUs=1' 'TimeLimit=02:00:00' 'mem=12G' 'gres/gpu:nvidia_a100_80gb_pcie_1g.10gb=1'; do
  [[ "$DETAIL" == *"$REQUIRED"* ]] || { echo "RESOURCE_MISMATCH job=$JOB missing=$REQUIRED" >&2; exit 3; }
done
echo "RESOURCE_PASS $JOB"
if [[ ${2:-} == --apply ]]; then scontrol release "$JOB"; echo "RELEASED $JOB"; else echo 'PREVIEW_ONLY: no jobs released'; fi
