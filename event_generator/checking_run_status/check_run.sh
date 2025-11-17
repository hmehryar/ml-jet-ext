#!/bin/bash
#SBATCH -J nb2py-run
#SBATCH -o /wsu/home/gy/gy40/gy4065/ml-jet-ext/logs/nb2py_%j.out
#SBATCH -e /wsu/home/gy/gy40/gy4065/ml-jet-ext/logs/nb2py_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -q primary
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 00:3:00

set -euo pipefail

# ---- USER SETTINGS (your real paths) ----
REPO_DIR="/wsu/home/gy/gy40/gy4065/ml-jet-ext"     # repo root containing the notebook
NB_PATH="$REPO_DIR/utils/check_run.ipynb"                # the notebook you built
OUT_PY="$REPO_DIR/utils/check_run.py"              # where to write the converted script
BASE_DIR="/rs/rs_grp_mljet/ml-jet-ext/simulation_results/"  # base dir of all runs (not used here, just FYI)
ENV_NAME="pytorch"                                  # conda env to use

# ---- prep dirs ----
mkdir -p "$REPO_DIR/utils" "$REPO_DIR/logs"

# ---- convert notebook -> script ----
jupyter nbconvert --to python "$NB_PATH"

# ---- conda env activation ----
# Try common Conda locations; adjust if your Conda is elsewhere
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  # Fallback to bashrc if your site initializes conda there
  source "$HOME/.bashrc" || true
fi

conda activate "$ENV_NAME"

# ---- run the script (no args) ----
# Ensure your check_run.py has an __main__ block that kicks off the batch scan.
cd "$REPO_DIR/utils"
python3 check_run.py

echo "[DONE] Converted notebook and executed check_run.py"
