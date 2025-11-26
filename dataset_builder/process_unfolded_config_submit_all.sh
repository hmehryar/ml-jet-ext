#!/usr/bin/env bash
set -euo pipefail


# --- locations ---
SRC="process_unfolded_config.py"
SBATCH_FILE="process_unfolded_config_dir.sbatch"

# --- sanity checks ---
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found in $(pwd)"; exit 1; }
[[ -f "$SBATCH_FILE" ]] || { echo "ERROR: $SBATCH_FILE not found in $(pwd)"; exit 1; }


# --- logs dir ---
mkdir -p logs


# Submit from the folder where ./jet_unfolding and unfold_dir.sbatch live
# DIRS=(
# c0005_alpha0p20_q_1p0 c0005_alpha0p20_q_1p5
# )
DIRS=(
c0005_alpha0p20_q_1p0  c0005_alpha0p20_q_1p5  c0005_alpha0p20_q_2p0  c0005_alpha0p20_q_2p5  c0005_alpha0p20_q_3p0
c0005_alpha0p25_q_1p0  c0005_alpha0p25_q_1p5  c0005_alpha0p25_q_2p0  c0005_alpha0p25_q_2p5  c0005_alpha0p25_q_3p0
c0005_alpha0p30_q_1p0  c0005_alpha0p30_q_1p5  c0005_alpha0p30_q_2p0  c0005_alpha0p30_q_2p5  c0005_alpha0p30_q_3p0
c0005_alpha0p35_q_1p0  c0005_alpha0p35_q_1p5  c0005_alpha0p35_q_2p0  c0005_alpha0p35_q_2p5  c0005_alpha0p35_q_3p0
c0005_alpha0p40_q_1p0  c0005_alpha0p40_q_1p5  c0005_alpha0p40_q_2p0  c0005_alpha0p40_q_2p5  c0005_alpha0p40_q_3p0
c0005_alpha0p45_q_1p0  c0005_alpha0p45_q_1p5  c0005_alpha0p45_q_2p0  c0005_alpha0p45_q_2p5  c0005_alpha0p45_q_3p0
c0005_alpha0p50_q_1p0  c0005_alpha0p50_q_1p5  c0005_alpha0p50_q_2p0  c0005_alpha0p50_q_2p5  c0005_alpha0p50_q_3p0
)

for d in "${DIRS[@]}"; do
  echo "Submitting ${d}..."
  sbatch \
    --export=RUN_DIR="$d" \
    -J "${d}" \
    -o "logs/process_unfolded_config_dir-${d}-%j.out" \
    -e "logs/process_unfolded_config_dir-${d}-%j.err" \
    process_unfolded_config_dir.sbatch
done

