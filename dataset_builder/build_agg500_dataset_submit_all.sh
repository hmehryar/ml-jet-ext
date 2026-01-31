#!/usr/bin/env bash
set -euo pipefail

SRC="03_build_agg500_dataset.py"
SBATCH_FILE="build_agg500_dataset_dir.sbatch"

# sanity checks
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found in $(pwd)"; exit 1; }
[[ -f "$SBATCH_FILE" ]] || { echo "ERROR: $SBATCH_FILE not found in $(pwd)"; exit 1; }

mkdir -p logs

# Same DIRS list as process_unfolded_config_submit_all.sh
DIRS=(
# # c0005_alpha0p20_q_1p0  c0005_alpha0p20_q_1p5  c0005_alpha0p20_q_2p0  c0005_alpha0p20_q_2p5  c0005_alpha0p20_q_3p0
# # c0005_alpha0p25_q_1p0  c0005_alpha0p25_q_1p5  
c0005_alpha0p25_q_2p0  
# # c0005_alpha0p25_q_2p5  c0005_alpha0p25_q_3p0
# # c0005_alpha0p30_q_1p0  c0005_alpha0p30_q_1p5  
# c0005_alpha0p30_q_2p0  c0005_alpha0p30_q_2p5  c0005_alpha0p30_q_3p0
# # c0005_alpha0p35_q_1p0
# c0005_alpha0p35_q_1p5  c0005_alpha0p35_q_2p0  c0005_alpha0p35_q_2p5  c0005_alpha0p35_q_3p0
# c0005_alpha0p40_q_1p0  c0005_alpha0p40_q_1p5  c0005_alpha0p40_q_2p0  c0005_alpha0p40_q_2p5  c0005_alpha0p40_q_3p0
# c0005_alpha0p45_q_1p0  c0005_alpha0p45_q_1p5  
# # c0005_alpha0p45_q_2p0  
# c0005_alpha0p45_q_2p5  c0005_alpha0p45_q_3p0
# c0005_alpha0p50_q_1p0  c0005_alpha0p50_q_1p5  
# c0005_alpha0p50_q_2p0  c0005_alpha0p50_q_2p5  c0005_alpha0p50_q_3p0
)

for d in "${DIRS[@]}"; do
  echo "Submitting agg500 for ${d}..."
  sbatch \
    --export=RUN_DIR="$d" \
    -J "agg500_${d}" \
    -o "logs/build_agg500_dataset_dir-${d}-%j.out" \
    -e "logs/build_agg500_dataset_dir-${d}-%j.err" \
    "${SBATCH_FILE}"
done
echo "Submitted all agg500 dataset jobs."