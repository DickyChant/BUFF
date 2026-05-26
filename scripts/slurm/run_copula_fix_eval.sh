#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_copula_fix
#SBATCH --output=slurm_copula_fix_%j.out
#SBATCH --error=slurm_copula_fix_%j.err
#SBATCH --export=ALL
#
# Apply the targeted (d_23, d_2) Gaussian-copula correction to existing
# flowBDT samples and re-measure:
#
#     - bootstrap discriminator AUC + per-feature AUC
#     - pair-feature AUC (12x12) -- to confirm (d_23, d_2) drops and
#       other pairs are not hurt by the correction
#     - FPD + KPD with truth-baseline floor
#
# Inputs are the *existing* generated_samples.npy from two prior runs:
#     baseline       : depth=4, n_est=100, per-feature
#     hicap+strip    : depth=6, n_est=200, multi-output, --strip-derived
#
# The script first runs copula_fix_pair.py to produce corrected samples,
# then runs the standard AUC + pair-AUC + FPD/KPD eval pipeline on them.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

REAL="${BUFF_REAL:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
BASELINE_GEN="${BUFF_BASELINE_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_t_hlv12/generated_samples.npy}"
HICAP_GEN="${BUFF_HICAP_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_hlv/jetnet_hlv_hicap_d6n200_strip/generated_samples.npy}"

ROOT="${BUFF_RESULTS_DIR}/jetnet_hlv_copula"
mkdir -p "${ROOT}"

BASELINE_CORR="${ROOT}/baseline_corrected.npy"
HICAP_CORR="${ROOT}/hicap_strip_corrected.npy"

# ---- Stage 1: apply copula correction to both gens ------------------------
echo
echo ">>> [1/4] Copula correction on (d_23, d_2) for baseline"
run_py -m BUFF.scripts.copula_fix_pair \
  --real "${REAL}" --gen "${BASELINE_GEN}" --out "${BASELINE_CORR}" \
  --feat-i d_23 --feat-j d_2

echo
echo ">>> [2/4] Copula correction on (d_23, d_2) for hicap+strip"
run_py -m BUFF.scripts.copula_fix_pair \
  --real "${REAL}" --gen "${HICAP_GEN}" --out "${HICAP_CORR}" \
  --feat-i d_23 --feat-j d_2

# ---- Stage 2: AUC bootstrap + per-feature on both corrected samples ------
echo
echo ">>> [3a/4] AUC eval on baseline_corrected"
run_py -m BUFF.scripts.run_jetnet_hlv_auc \
  --real "${REAL}" --gen "${BASELINE_CORR}" \
  --out-dir "${ROOT}/baseline_corrected_auc" \
  --n-bootstrap 20

echo
echo ">>> [3b/4] AUC eval on hicap_strip_corrected"
run_py -m BUFF.scripts.run_jetnet_hlv_auc \
  --real "${REAL}" --gen "${HICAP_CORR}" \
  --out-dir "${ROOT}/hicap_strip_corrected_auc" \
  --n-bootstrap 20

# ---- Stage 3: Pair-feature AUC 12x12 (just on the more interesting one) --
echo
echo ">>> [3c/4] Pair-feature AUC 12x12 on hicap_strip_corrected"
run_py -m BUFF.scripts.run_jetnet_hlv_pair_auc \
  --real "${REAL}" --gen "${HICAP_CORR}" \
  --out-dir "${ROOT}/hicap_strip_corrected_pairs" \
  --n-events 30000

echo
echo ">>> [3d/4] Pair-feature AUC 12x12 on baseline_corrected"
run_py -m BUFF.scripts.run_jetnet_hlv_pair_auc \
  --real "${REAL}" --gen "${BASELINE_CORR}" \
  --out-dir "${ROOT}/baseline_corrected_pairs" \
  --n-events 30000

# ---- Stage 4: FPD/KPD on both corrected samples --------------------------
echo
echo ">>> [4a/4] FPD + KPD on baseline_corrected"
run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
  --real "${REAL}" --gen "${BASELINE_CORR}" --mode hlv \
  --out-dir "${ROOT}/baseline_corrected_fpd"

echo
echo ">>> [4b/4] FPD + KPD on hicap_strip_corrected"
run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
  --real "${REAL}" --gen "${HICAP_CORR}" --mode hlv \
  --out-dir "${ROOT}/hicap_strip_corrected_fpd"

echo
echo "COPULA FIX + EVAL DONE $(date -Is)"
echo "  Root        : ${ROOT}/"
echo "  Corrected   : ${BASELINE_CORR}"
echo "                ${HICAP_CORR}"
echo "  Eval results: ${ROOT}/*_auc/auc_bootstrap.json"
echo "                ${ROOT}/*_pairs/pair_auc.json"
echo "                ${ROOT}/*_fpd/fpd_kpd_results.json"
