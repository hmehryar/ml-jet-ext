#!/usr/bin/env bash
#SBATCH --job-name=T04_10_single
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/slurm_logs/T04_10_%j.out
#SBATCH --error=/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/slurm_logs/T04_10_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
# Preserve the immutable source snapshot during Python imports.
export PYTHONDONTWRITEBYTECODE=1
: "${T04_10_PREPARED_RUN:?Must export the immutable prepared-run directory}"
: "${T04_10_MODE:?Must export fresh or resume}"
[[ "$T04_10_MODE" == fresh || "$T04_10_MODE" == resume ]] || exit 2

# Revalidate *before* importing large training dependencies.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
RUN="$(realpath "$T04_10_PREPARED_RUN")"
# Slurm executes a spool copy, so BASH_SOURCE[0] is NOT the repository path.
: "${T04_10_REPO:?Must export source repository from the submission script}"
REPO="$(realpath "$T04_10_REPO")"
[[ -f "$REPO/tools/t04_10_stage5a_guard.py" ]] || { echo 'Missing T04-10 job guard' >&2; exit 2; }
PYTHONPATH="$REPO" python "$REPO/tools/t04_10_stage5a_guard.py" \
  --run "$RUN" --mode "$T04_10_MODE"

printf 'T04_10_START job=%s host=%s mode=%s run=%s cpus=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$T04_10_MODE" "$RUN" "${SLURM_CPUS_PER_TASK:-unknown}"
nvidia-smi -L || true
cd "$RUN/source"
export PYTHONPATH="$RUN/source"
exec python "$RUN/source/train_utils/t04_10_runner.py" \
  --prepared-run "$RUN" --mode "$T04_10_MODE"
