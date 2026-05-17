#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_fig8_unfold
#SBATCH --output=slurm_fig8_unfold_%j.out
#SBATCH --error=slurm_fig8_unfold_%j.err
#SBATCH --export=ALL
#
# Reproduce Figure 8 of the BUFF paper: unfolding comparison with Flow-Diffu
# (Gaussian prior) and Flow-OT (detector-level prior).
#
# Pipeline:
#   1. Fetch + preprocess omnifold Pythia26 dataset (6 features)
#   2. Train flowBDT Flow-Diffu (Gaussian -> particle-level)
#   3. Train flowBDT Flow-OT   (detector-level -> particle-level)
#   4. Plot Fig 8 (fold_comp.pdf, fold_comp_compact.pdf)
#
# Skip stages via env:
#   BUFF_SKIP_DATA=1   reuse cached features
#   BUFF_SKIP_DIFFU=1  reuse trained Flow-Diffu
#   BUFF_SKIP_OT=1     reuse trained Flow-OT
#   BUFF_SKIP_PLOT=1   stop before plotting

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

SAMPLE="${BUFF_FIG8_SAMPLE:-Pythia26}"
TAG="${BUFF_FIG8_TAG:-pythia26}"
N_EVENTS="${BUFF_FIG8_N_EVENTS:-300000}"
N_KEEP="${BUFF_FIG8_N_KEEP:-0}"  # 0 = all that passed selection

DATA_DIR="${BUFF_RESULTS_DIR}/omnifold/data"
WORK="${BUFF_RESULTS_DIR}/omnifold/${TAG}"
GEN_NPY="${DATA_DIR}/${TAG}_gen_features.npy"
SIM_NPY="${DATA_DIR}/${TAG}_sim_features.npy"
DIFFU_DIR="${WORK}/flow_diffu"
OT_DIR="${WORK}/flow_ot"
PLOT_OUT="${WORK}/fold_comp.pdf"
PLOT_COMPACT="${WORK}/fold_comp_compact.pdf"

mkdir -p "${DATA_DIR}" "${WORK}"

# --- Stage 1: build dataset ----------------------------------------------
if [[ "${BUFF_SKIP_DATA:-0}" != "1" ]]; then
  echo ">>> [1/4] Build omnifold features"
  run_py -m BUFF.scripts.build_omnifold_unfolding \
    --sample "${SAMPLE}" \
    --n-events "${N_EVENTS}" \
    --cache-dir "${BUFF_RESULTS_DIR}/data/omnifold/cache" \
    --out-dir "${DATA_DIR}" \
    --tag "${TAG}"
else
  echo ">>> [1/4] Build SKIPPED  (reusing ${GEN_NPY}, ${SIM_NPY})"
fi

# --- Stage 2: Flow-Diffu  (Gaussian prior -> gen) ------------------------
if [[ "${BUFF_SKIP_DIFFU:-0}" != "1" ]]; then
  echo
  echo ">>> [2/4] Train Flow-Diffu (Gaussian prior)"
  mkdir -p "${DIFFU_DIR}"
  run_py -m BUFF.runner.train_and_sample \
    --data "${GEN_NPY}" \
    --output-dir "${DIFFU_DIR}" \
    --n-timesteps 30 --duplicate-k 20 --multi-output \
    --flow-type icfm --sigma 0.0 \
    --solver dopri5 --solver-steps 30 \
    --max-depth 4 --n-estimators 100 --eta 0.1 \
    --n-jobs 1 --n-threads "${OMP_NUM_THREADS}"
else
  echo ">>> [2/4] Flow-Diffu SKIPPED"
fi

# --- Stage 3: Flow-OT   (sim prior -> gen) -------------------------------
if [[ "${BUFF_SKIP_OT:-0}" != "1" ]]; then
  echo
  echo ">>> [3/4] Train Flow-OT (sim prior; index-matched pairs)"
  mkdir -p "${OT_DIR}"
  run_py -m BUFF.runner.train_and_sample \
    --data "${GEN_NPY}" \
    --source-data "${SIM_NPY}" \
    --output-dir "${OT_DIR}" \
    --n-timesteps 30 --duplicate-k 20 --multi-output \
    --flow-type icfm --sigma 0.0 \
    --solver dopri5 --solver-steps 30 \
    --max-depth 4 --n-estimators 100 --eta 0.1 \
    --n-jobs 1 --n-threads "${OMP_NUM_THREADS}"
else
  echo ">>> [3/4] Flow-OT SKIPPED"
fi

# --- Stage 4: plot Fig 8 -------------------------------------------------
if [[ "${BUFF_SKIP_PLOT:-0}" != "1" ]]; then
  echo
  echo ">>> [4/4] Plot Fig 8"
  run_py -m BUFF.scripts.plot_fig8 \
    --gen "${GEN_NPY}" \
    --sim "${SIM_NPY}" \
    --flow-diffu "${DIFFU_DIR}/generated_samples.npy" \
    --flow-ot   "${OT_DIR}/generated_samples.npy" \
    --out "${PLOT_OUT}" \
    --ratio-out "${PLOT_COMPACT}"
else
  echo ">>> [4/4] Plot SKIPPED"
fi

echo
echo "FIG8 PIPELINE DONE $(date -Is)"
echo "  Data    : ${GEN_NPY}, ${SIM_NPY}"
echo "  Diffu   : ${DIFFU_DIR}/generated_samples.npy"
echo "  Flow-OT : ${OT_DIR}/generated_samples.npy"
echo "  PDF     : ${PLOT_OUT}"
echo "  Compact : ${PLOT_COMPACT}"
