#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_jetnet_hlv_hicap
#SBATCH --output=slurm_jetnet_hlv_hicap_%j.out
#SBATCH --error=slurm_jetnet_hlv_hicap_%j.err
#SBATCH --export=ALL
#
# Higher-capacity 12-feature JetNet HLV retrain aimed at reducing the
# discriminator AUC.  The baseline run (depth=4, n_est=100, no strip-derived,
# no multi-output) produced AUC ~= 0.9994 -- essentially perfect separation.
#
# Diagnosis (login-node quick test) showed:
#   - 3% of generated events sit at feature-range boundaries (min-max clip)
#   - Post-hoc Cholesky correction makes things WORSE (AUC -> 1.0000) because
#     it shifts marginals
#   - Adding tiny noise or dropping boundary events does not move the needle
# So the AUC is driven by genuine joint-distribution gaps, not artifacts.
#
# This retrain uses:
#   --strip-derived    train only 9 independent features, derive tau21,
#                      tau32, d2_obs analytically post-generation. Eliminates
#                      derived-feature inconsistency that the discriminator
#                      can detect.
#   --multi-output     one multi-output XGBoost per timestep (joint outputs)
#   --max-depth 6      bigger trees
#   --n-estimators 200 more boosting rounds
#   --eta 0.05         slower learning rate, same as 30x3 hicap precedent
#
# Expected training time: ~1.5-3h (HLV is 12-D; smaller than 30x3's 90-D).

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

DATA="${BUFF_HLV_DATA:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
TAG="${BUFF_HLV_TAG:-jetnet_hlv_hicap_d6n200_strip}"

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

mkdir -p "${WORK}"

# --- Stage 1: hicap + strip-derived + multi-output train + sample ---------
if [[ "${BUFF_SKIP_TRAIN:-0}" != "1" ]]; then
  echo
  echo ">>> [1/2] Train flowBDT (strip-derived, multi-output, depth=6, n_est=200, eta=0.05)"
  EXTRA="${BUFF_HLV_TRAIN_ARGS:---n-timesteps 30 --duplicate-k 20 --multi-output --strip-derived --flow-type icfm --sigma 0.0 --solver dopri5 --solver-steps 30 --max-depth 6 --n-estimators 200 --eta 0.05}"
  # shellcheck disable=SC2086
  run_py -m BUFF.runner.train_and_sample \
    --data "${DATA}" \
    --output-dir "${WORK}" \
    --n-jobs "${BUFF_HLV_NJOBS:-1}" \
    --n-threads "${BUFF_HLV_NTHREADS:-128}" \
    ${EXTRA}
else
  echo ">>> [1/2] Train SKIPPED  (reusing ${GEN_NPY})"
fi

# --- Stage 2: evaluate (AUC + per-feature + diagnostics) ------------------
if [[ "${BUFF_SKIP_EVAL:-0}" != "1" ]]; then
  echo
  echo ">>> [2/2] AUC bootstrap evaluation + per-feature breakdown"
  mkdir -p "${EVAL_OUT}"
  run_py -m BUFF.scripts.run_jetnet_hlv_auc \
    --real "${DATA}" \
    --gen "${GEN_NPY}" \
    --out-dir "${EVAL_OUT}" \
    --n-bootstrap "${BUFF_HLV_NBOOT:-20}"

  echo
  echo ">>> [2b/2] Full W1 + sep-power bootstrap eval (existing pipeline)"
  run_py -m BUFF.evaluation.run_jetnet_eval \
    --real "${DATA}" \
    --gen "${GEN_NPY}" \
    --n-bootstrap "${BUFF_HLV_NBOOT:-100}" \
    --output-dir "${EVAL_OUT}"
else
  echo ">>> [2/2] Eval SKIPPED"
fi

echo
echo "HLV HICAP PIPELINE DONE $(date -Is)"
echo "  Generated  : ${GEN_NPY}"
echo "  Eval dir   : ${EVAL_OUT}/"
echo "  AUC json   : ${EVAL_OUT}/auc_bootstrap.json"
