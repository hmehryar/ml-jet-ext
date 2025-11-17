#!/usr/bin/env bash
set -euo pipefail


# --- locations ---
FASTJET_CFG="/wsu/el7/gnu7/fastjet/3.4.0/bin/fastjet-config"
SRC="jet_unfolding_50_250gev.cc"
BIN="./jet_unfolding"
SBATCH_FILE="jet_unfolding_dir.sbatch"

# --- sanity checks ---
[[ -x "$FASTJET_CFG" ]] || { echo "ERROR: fastjet-config not found at $FASTJET_CFG"; exit 1; }
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found in $(pwd)"; exit 1; }
[[ -f "$SBATCH_FILE" ]] || { echo "ERROR: $SBATCH_FILE not found in $(pwd)"; exit 1; }

# --- compile (only if needed) ---
if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  echo "[build] Compiling $SRC -> $BIN ..."
  g++ -std=c++11 "$SRC" -O3 -o "$BIN" $($FASTJET_CFG --cxxflags --libs --plugins)
  echo "[build] Done."
else
  echo "[build] Up-to-date: $BIN"
fi

# --- logs dir ---
mkdir -p logs


# Submit from the folder where ./jet_unfolding and unfold_dir.sbatch live
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
    -o "logs/jet_unfolding_50_250GeV/jet-unfolding-${d}-%j.out" \
    -e "logs/jet_unfolding_50_250GeV/jet-unfolding-${d}-%j.err" \
    jet_unfolding_dir.sbatch
done

