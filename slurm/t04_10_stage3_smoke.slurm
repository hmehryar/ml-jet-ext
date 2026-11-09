#!/usr/bin/env bash
#SBATCH --job-name=T04-10-GPU-SMOKE
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3_gpu_smoke/logs/%x_%j.out
#SBATCH --error=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage3_gpu_smoke/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "=========================================="
echo "T04-10 Stage 3: Real GPU Integration"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"
echo "Started: $(date)"

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA unavailable"

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

assert torch.cuda.is_bf16_supported(), "BF16 unsupported"
PY

for model in clip siglip; do
    for task in alpha_only q0_only; do

        config="experiments/t04/config/t04_10_stage3_smoke/t04_10_${model}_${task}_seed42.yml"

        echo "=========================================="
        echo "Model: $model"
        echo "Task: $task"
        echo "Configuration: $config"
        echo "=========================================="

        python tools/t04_10_real_cuda_smoke.py \
            --config "$config"

        echo "PASS: $model / $task"
    done
done

echo "=========================================="
echo "ALL FOUR CUDA SMOKE TESTS PASSED"
echo "=========================================="
date
