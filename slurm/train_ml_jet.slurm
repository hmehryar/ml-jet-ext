#!/bin/bash
#SBATCH --job-name=jext-vit-f0-e400-lr3-lrp100

#SBATCH --partition=ecscp
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=504:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=express

##SBATCH --constraint=v100
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=1
##SBATCH --mem=60G
##SBATCH --time=504:00:00
##SBATCH --nodes=1
##SBATCH --ntasks=4
##SBATCH --qos=gpu


nvidia-smi

CONFIG_PATH="$1"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "❌ Config file not found: $CONFIG_PATH"
  exit 1
fi

# === Extract YAML Fields Using Python ===
get_yaml_value() {
  python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['$1'])"
}
MODEL_TAG=$(get_yaml_value model_tag)
BATCH_SIZE=$(get_yaml_value batch_size)
EPOCHS=$(get_yaml_value epochs)
LEARNING_RATE=$(get_yaml_value learning_rate)
FOLD=$(get_yaml_value fold_index)

RUN_TAG="${MODEL_TAG}_bs${BATCH_SIZE}_ep${EPOCHS}_lr$(printf %.0e $LEARNING_RATE)_fold${FOLD}"
LOGS_DIR="logs"
mkdir -p $LOGS_DIR
LOG_FILE="${LOGS_DIR}/train_${RUN_TAG}_${SLURM_JOB_ID}.out"
ERR_FILE="${LOGS_DIR}/train_${RUN_TAG}_${SLURM_JOB_ID}.err"
exec > $LOG_FILE 2> $ERR_FILE

module purge
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh
conda init
# conda activate tensorflow-gpu-v2.8
conda activate pytorch

START_TIME=$(date +%s)
jupyter nbconvert --to script train.ipynb --output train_${RUN_TAG}_${SLURM_JOB_ID}
python train_${RUN_TAG}_${SLURM_JOB_ID}.py --config "$CONFIG_PATH"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "[INFO] Training completed in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"
echo "[INFO] SLURM Job finished on: $(date)"
