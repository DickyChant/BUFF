#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_nn_flow_auc
#SBATCH --output=slurm_nn_flow_auc_%j.out
#SBATCH --error=slurm_nn_flow_auc_%j.err
#SBATCH --export=ALL
#
# NN-flow baseline for the JetNet HLV discriminator-AUC comparison.
# Identical objective (ICFM with sigma=0) and dataset (12-feature top HLV)
# as the BUFF-BDT runs; only the regressor backbone differs.  Trains an
# MLPVelocity and reports the bootstrap discriminator AUC + per-feature
# breakdown, so we can place it side-by-side with:
#
#     BUFF baseline (per-feature, depth=4, n_est=100) -- AUC ~= 0.9994
#     BUFF hicap+strip (multi-output, depth=6, n_est=200, strip-derived)
#                                                       -- AUC ~= 0.9974
#
# Defaults are sized to roughly match the BUFF training cost so the
# comparison is fair on compute.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

DATA="${BUFF_NN_DATA:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
TAG="${BUFF_NN_TAG:-jetnet_hlv_nn_h128_d4_e30}"

WORK="${BUFF_RESULTS_DIR}/jetnet_hlv_nn/${TAG}"
mkdir -p "${WORK}"

HIDDEN="${BUFF_NN_HIDDEN:-128}"
DEPTH="${BUFF_NN_DEPTH:-4}"
EPOCHS="${BUFF_NN_EPOCHS:-30}"
BATCH="${BUFF_NN_BATCH:-1024}"
NSTEPS="${BUFF_NN_NSTEPS:-30}"
NBOOT="${BUFF_NN_NBOOT:-20}"

echo "Data       : ${DATA}"
echo "Tag        : ${TAG}"
echo "Work dir   : ${WORK}"
echo "Model      : MLP hidden=${HIDDEN} depth=${DEPTH} epochs=${EPOCHS} batch=${BATCH}"

run_py -m BUFF.scripts.run_neural_flow_auc \
  --data "${DATA}" \
  --out-dir "${WORK}" \
  --hidden "${HIDDEN}" \
  --depth "${DEPTH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --n-steps "${NSTEPS}" \
  --n-bootstrap "${NBOOT}" \
  --n-threads "${BUFF_NN_NTHREADS:-${OMP_NUM_THREADS:-16}}"

echo
echo "NN FLOW AUC PIPELINE DONE $(date -Is)"
echo "  AUC json : ${WORK}/nn_flow_auc.json"
echo "  Gen npy  : ${WORK}/nn_generated_samples.npy"
