#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_lgbm_residual
#SBATCH --output=slurm_lgbm_residual_%j.out
#SBATCH --error=slurm_lgbm_residual_%j.err
#SBATCH --export=ALL
#
# LightGBM linear-tree + two-pass residual on JetNet HLV.
#
# Combines two structural innovations in one run:
#   1. LightGBM with linear_tree=True.  Each leaf fits a linear regression
#      instead of a constant, so the velocity field v(x, t) is piecewise-
#      LINEAR in x rather than piecewise-constant.  Adaptive ODE solvers
#      (DOPRI5, Midpoint) need some smoothness to choose step sizes; this
#      should let them take larger steps with less integration error.
#   2. Two-pass residual training.  Pass-1 per-feature regressors give a
#      baseline v_hat; Pass-2 regressors take input [xt, v_hat] and predict
#      the residual.  Pass-2 sees what other features predicted, recovering
#      joint correlations that single-output LightGBM trees miss (LightGBM
#      has no vector-leaf multi-output analog of XGBoost's
#      multi_strategy="multi_output_tree").
#
# Both fixes target the discriminator-AUC ceiling (~ 0.997 in the
# XGBoost-multi-output run).  Expected training time: ~6-10h (12h budget).

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

DATA="${BUFF_HLV_DATA:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
TAG="${BUFF_LGBM_TAG:-jetnet_hlv_lgbm_lt_residual_d6n200}"

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
echo "Config     : LightGBM linear_tree + 2-pass residual + strip-derived,"
echo "             depth=6, n_est=200, eta=0.05, K=20, n_t=30"

mkdir -p "${WORK}"

# --- Stage 1: train + sample -----------------------------------------------
if [[ "${BUFF_SKIP_TRAIN:-0}" != "1" ]]; then
  echo
  echo ">>> [1/2] Train flowBDT (LightGBM linear_tree + residual + strip)"
  EXTRA="${BUFF_LGBM_TRAIN_ARGS:---n-timesteps 30 --duplicate-k 20 --strip-derived --residual --regressor lightgbm --linear-tree --flow-type icfm --sigma 0.0 --solver dopri5 --solver-steps 30 --max-depth 6 --n-estimators 200 --eta 0.05}"
  # shellcheck disable=SC2086
  run_py -m BUFF.runner.train_and_sample \
    --data "${DATA}" \
    --output-dir "${WORK}" \
    --n-jobs "${BUFF_LGBM_NJOBS:-1}" \
    --n-threads "${BUFF_LGBM_NTHREADS:-128}" \
    ${EXTRA}
else
  echo ">>> [1/2] Train SKIPPED  (reusing ${GEN_NPY})"
fi

# --- Stage 2: evaluate (AUC + per-feature + W1 + FPD/KPD) ------------------
if [[ "${BUFF_SKIP_EVAL:-0}" != "1" ]]; then
  echo
  echo ">>> [2a/2] AUC bootstrap + per-feature breakdown"
  mkdir -p "${EVAL_OUT}"
  run_py -m BUFF.scripts.run_jetnet_hlv_auc \
    --real "${DATA}" \
    --gen "${GEN_NPY}" \
    --out-dir "${EVAL_OUT}" \
    --n-bootstrap "${BUFF_LGBM_NBOOT:-20}"

  echo
  echo ">>> [2b/2] FPD + KPD + truth-baseline floor"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${DATA}" \
    --gen "${GEN_NPY}" \
    --mode hlv \
    --out-dir "${EVAL_OUT}/fpd_kpd"

  echo
  echo ">>> [2c/2] Full W1 + sep-power bootstrap eval"
  run_py -m BUFF.evaluation.run_jetnet_eval \
    --real "${DATA}" \
    --gen "${GEN_NPY}" \
    --n-bootstrap "${BUFF_LGBM_NBOOT_W1:-100}" \
    --output-dir "${EVAL_OUT}"
else
  echo ">>> [2/2] Eval SKIPPED"
fi

echo
echo "LGBM RESIDUAL PIPELINE DONE $(date -Is)"
echo "  Generated  : ${GEN_NPY}"
echo "  Eval dir   : ${EVAL_OUT}/"
echo "  AUC json   : ${EVAL_OUT}/auc_bootstrap.json"
echo "  FPD/KPD    : ${EVAL_OUT}/fpd_kpd/fpd_kpd_results.json"
