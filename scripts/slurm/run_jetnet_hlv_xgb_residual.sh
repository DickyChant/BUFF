#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_xgb_residual
#SBATCH --output=slurm_xgb_residual_%j.out
#SBATCH --error=slurm_xgb_residual_%j.err
#SBATCH --export=ALL
#
# XGBoost (per-feature) + two-pass residual + strip-derived.
#
# Replaces the LightGBM+residual experiment now that LightGBM's linear_tree
# was confirmed no-op (empty leaf_coeff lists).  XGBoost backend means no
# linear-leaf benefit but the residual two-pass remains the lever.
#
# Pass 1: per-feature XGBoost regressors give a baseline v_hat
# Pass 2: per-feature XGBoost regressors take input [xt, v_hat] and predict
#         the residual.  Pass 2 sees what other features predicted,
#         recovering joint correlations that the per-feature MSE objective
#         loses.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

DATA="${BUFF_HLV_DATA:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
TAG="${BUFF_TAG:-jetnet_hlv_xgb_residual_d6n200}"

WORK="${BUFF_RESULTS_DIR}/jetnet_hlv/${TAG}"
GEN_NPY="${WORK}/generated_samples.npy"
EVAL_OUT="${WORK}/eval"

if [[ ! -f "${DATA}" ]]; then
  echo "ERROR: HLV data not found at ${DATA}" >&2
  exit 1
fi

echo "Data       : ${DATA}"
echo "Tag        : ${TAG}"
echo "Work dir   : ${WORK}"
echo "Config     : XGBoost per-feature + 2-pass residual + strip-derived,"
echo "             depth=6, n_est=200, eta=0.05, K=20, n_t=30"

mkdir -p "${WORK}"

# --- Stage 1: train + sample ---
if [[ "${BUFF_SKIP_TRAIN:-0}" != "1" ]]; then
  echo
  echo ">>> [1/2] Train flowBDT (XGBoost + residual + strip)"
  EXTRA="${BUFF_TRAIN_ARGS:---n-timesteps 30 --duplicate-k 20 --strip-derived --residual --regressor xgboost --flow-type icfm --sigma 0.0 --solver dopri5 --solver-steps 30 --max-depth 6 --n-estimators 200 --eta 0.05}"
  # shellcheck disable=SC2086
  run_py -m BUFF.runner.train_and_sample \
    --data "${DATA}" \
    --output-dir "${WORK}" \
    --n-jobs "${BUFF_NJOBS:-1}" \
    --n-threads "${BUFF_NTHREADS:-128}" \
    ${EXTRA}
else
  echo ">>> [1/2] Train SKIPPED  (reusing ${GEN_NPY})"
fi

# --- Stage 2: AUC + FPD/KPD + pair-AUC + W1 ---
if [[ "${BUFF_SKIP_EVAL:-0}" != "1" ]]; then
  mkdir -p "${EVAL_OUT}"

  echo
  echo ">>> [2a/2] AUC bootstrap + per-feature breakdown"
  run_py -m BUFF.scripts.run_jetnet_hlv_auc \
    --real "${DATA}" --gen "${GEN_NPY}" \
    --out-dir "${EVAL_OUT}" --n-bootstrap "${BUFF_NBOOT:-20}"

  echo
  echo ">>> [2b/2] FPD + KPD + truth-baseline"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${DATA}" --gen "${GEN_NPY}" --mode hlv \
    --out-dir "${EVAL_OUT}/fpd_kpd"

  echo
  echo ">>> [2c/2] Pair-feature AUC (12x12 heatmap)"
  run_py -m BUFF.scripts.run_jetnet_hlv_pair_auc \
    --real "${DATA}" --gen "${GEN_NPY}" \
    --out-dir "${EVAL_OUT}/pairs" --n-events 30000

  echo
  echo ">>> [2d/2] Full W1 + sep-power eval"
  run_py -m BUFF.evaluation.run_jetnet_eval \
    --real "${DATA}" --gen "${GEN_NPY}" \
    --n-bootstrap "${BUFF_NBOOT_W1:-100}" --output-dir "${EVAL_OUT}"
else
  echo ">>> [2/2] Eval SKIPPED"
fi

echo
echo "XGB RESIDUAL PIPELINE DONE $(date -Is)"
echo "  Generated : ${GEN_NPY}"
echo "  Eval dir  : ${EVAL_OUT}/"
