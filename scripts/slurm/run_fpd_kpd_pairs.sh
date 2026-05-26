#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_fpd_kpd_pairs
#SBATCH --output=slurm_fpd_kpd_pairs_%j.out
#SBATCH --error=slurm_fpd_kpd_pairs_%j.err
#SBATCH --export=ALL
#
# Two parallel diagnostics on the existing JetNet HLV generated samples
# (NO retraining):
#
#   1. FPD + KPD (JetNet-paper finite-sample-bias-corrected metrics) with
#      truth-baseline floor.  Run on both the baseline BUFF samples and
#      the hicap+strip retrain, so we can quote both.
#
#   2. Pair-feature discriminator AUC (12x12 matrix) on the hicap+strip
#      samples, to localize WHERE the joint-correlation gap lives.
#
# Optionally also runs FPD/KPD on the NN-flow samples if they have landed
# at the expected path by the time this job starts.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

REAL="${BUFF_REAL:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
BASELINE_GEN="${BUFF_BASELINE_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_t_hlv12/generated_samples.npy}"
HICAP_GEN="${BUFF_HICAP_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_hlv/jetnet_hlv_hicap_d6n200_strip/generated_samples.npy}"
NN_GEN="${BUFF_NN_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_hlv_nn/jetnet_hlv_nn_h128_d4_e30/nn_generated_samples.npy}"

OUT_DIR="${BUFF_RESULTS_DIR}/jetnet_hlv_fpd_pairs"
mkdir -p "${OUT_DIR}"

echo "Real        : ${REAL}"
echo "Baseline gen: ${BASELINE_GEN}"
echo "Hicap gen   : ${HICAP_GEN}"
echo "NN gen      : ${NN_GEN}  (optional)"
echo "Out         : ${OUT_DIR}"

# ---- FPD + KPD: baseline ----
echo
echo ">>> FPD + KPD on BUFF baseline (d=4, n=100)"
run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
  --real "${REAL}" --gen "${BASELINE_GEN}" --mode hlv \
  --out-dir "${OUT_DIR}/baseline"

# ---- FPD + KPD: hicap+strip ----
echo
echo ">>> FPD + KPD on BUFF hicap+strip (multi-output, d=6, n=200, eta=0.05)"
run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
  --real "${REAL}" --gen "${HICAP_GEN}" --mode hlv \
  --out-dir "${OUT_DIR}/hicap_strip"

# ---- FPD + KPD: NN flow (only if it exists) ----
if [[ -f "${NN_GEN}" ]]; then
  echo
  echo ">>> FPD + KPD on NN-flow (MLP h=128 d=4)"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${REAL}" --gen "${NN_GEN}" --mode hlv \
    --out-dir "${OUT_DIR}/nn_flow"
else
  echo
  echo ">>> NN-flow samples not present yet, skipping ${NN_GEN}"
fi

# ---- Pair-feature AUC: hicap+strip ----
echo
echo ">>> Pair-feature AUC (12x12) on hicap+strip"
run_py -m BUFF.scripts.run_jetnet_hlv_pair_auc \
  --real "${REAL}" --gen "${HICAP_GEN}" \
  --out-dir "${OUT_DIR}/hicap_strip_pairs" \
  --n-events 30000

# ---- Pair-feature AUC: baseline (for direct contrast) ----
echo
echo ">>> Pair-feature AUC (12x12) on baseline"
run_py -m BUFF.scripts.run_jetnet_hlv_pair_auc \
  --real "${REAL}" --gen "${BASELINE_GEN}" \
  --out-dir "${OUT_DIR}/baseline_pairs" \
  --n-events 30000

echo
echo "FPD/KPD/PAIRS PIPELINE DONE $(date -Is)"
echo "  Out dir : ${OUT_DIR}/"
