#!/usr/bin/env bash
#SBATCH --job-name=T03_snapshot
#SBATCH --partition=ecscp
#SBATCH --qos=express
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Prepare with tools/run_t03.py --prepare-only BEFORE sbatch. The job imports
# only that snapshot; the checkout may subsequently change without affecting it.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this launcher within a Slurm allocation}"
RUN_DIR="${1:?Usage: sbatch slurm/train_t03_snapshot.slurm /absolute/prepared/run}"
PYTHON_BIN="${MLJET_PYTHON:-python}"
[[ "$RUN_DIR" = /* && -f "$RUN_DIR/source/train_t03_run.py" ]] || {
    echo "Expected an absolute prepared run directory" >&2; exit 1;
}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN_DIR/source"
cd "$RUN_DIR/source"
# Validate worker count against this allocation without importing torch.
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json, os, pathlib, sys
cfg = json.loads((pathlib.Path(sys.argv[1]) / 'source/config.resolved.json').read_text())
if cfg['profile'] != 'hpc':
    raise SystemExit('Slurm launcher requires a prepared --profile hpc run')
if cfg['num_workers'] > int(os.environ['SLURM_CPUS_PER_TASK']):
    raise SystemExit('DataLoader workers exceed allocated CPUs')
PY
exec "$PYTHON_BIN" -B -u "$RUN_DIR/source/train_t03_run.py" --prepared-run "$RUN_DIR"
