#!/usr/bin/env bash
#SBATCH --job-name=T04-10-RESUME
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3b_resume/logs/%x_%j.out
#SBATCH --error=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3b_resume/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3b_resume
: "${SLURM_JOB_ID:?Must execute inside a Slurm job}"
RUN="$BASE/job_${SLURM_JOB_ID}"
mkdir "$RUN"  # fail closed on duplicate or previous run
printf 'STAGE3B_JOB=%s COMMIT=%s NODE=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$(hostname)"
nvidia-smi
python - <<'PY'
import torch
print('CUDA:',torch.cuda.is_available(),'BF16:',torch.cuda.is_bf16_supported(),flush=True)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PY
for model in clip siglip; do
  for task in alpha_only q0_only; do
    config="experiments/t04/config/t04_10_stage3_smoke/t04_10_${model}_${task}_seed42.yml"
    echo "STAGE3B_START model=$model task=$task"
    python -m tools.t04_10_stage3b_resume --config "$config" --output-root "$RUN" \
      --train-samples 6 --val-samples 6
    echo "STAGE3B_PASS model=$model task=$task"
  done
done
echo 'STAGE3B_ALL_FOUR_PASS'
