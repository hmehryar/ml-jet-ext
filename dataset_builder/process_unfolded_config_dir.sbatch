#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -q primary

set -euo pipefail

# Expect RUN_DIR via --export (see submit script)
: "${RUN_DIR:?Set with --export=RUN_DIR=<dir>}"

# Adjust these if needed
BASE="/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/"
BIN="process_unfolded_config.py"   # run sbatch from the directory containing this binary

IN="${BASE}/${RUN_DIR}_unfolded"
OUT="/rs/rs_grp_mljet/ml-jet-ext/pTHat50-250_eCM5020_pTHat50-250_eCM5020_pT90-110_R0p4"

echo "[`date`] START RUN_DIR=${RUN_DIR}"
echo "[INFO] IN=${IN}"
echo "[INFO] OUT=${OUT}"

# jet_unfolding already prints the file count and result summary
python "${BIN}" --config-dir "${IN}" --output-root "${OUT}"

echo "[`date`] DONE RUN_DIR=${RUN_DIR}"
