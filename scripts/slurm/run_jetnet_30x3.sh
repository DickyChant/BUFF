#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_jetnet_30x3
#SBATCH --output=slurm_jetnet_30x3_%j.out
#SBATCH --error=slurm_jetnet_30x3_%j.err
#SBATCH --export=ALL
#
# JetNet 30x3 particle-level pipeline: build dataset, train+sample flowBDT,
# evaluate W1 metrics with bootstrap + truth baseline.
#
# Stages can be skipped individually:
#   BUFF_SKIP_BUILD=1   skip building the leading-30 npy
#   BUFF_SKIP_TRAIN=1   skip train_and_sample (assumes generated_samples.npy exists)
#   BUFF_SKIP_EVAL=1    skip the W1 evaluation
#
# Environment knobs (with defaults):
#   BUFF_JETNET_RAW         raw JetNet hdf5 (top jets)
#   BUFF_30X3_TAG           sub-dir tag inside ${BUFF_RESULTS_DIR}/jetnet_30x3/
#   BUFF_30X3_TRAIN_ARGS    override the entire training arg list

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

RAW="${BUFF_JETNET_RAW:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet_raw/t.hdf5}"
TAG="${BUFF_30X3_TAG:-t30x3}"

DATA_DIR="${BUFF_RESULTS_DIR}/jetnet_30x3/data"
WORK="${BUFF_RESULTS_DIR}/jetnet_30x3/${TAG}"
DATA_NPY="${DATA_DIR}/${TAG}_p30_kin.npy"
GEN_NPY="${WORK}/generated_samples.npy"
EVAL_OUT="${WORK}/eval"

if [[ ! -f "${RAW}" ]]; then
  echo "ERROR: raw JetNet hdf5 not found at ${RAW}" >&2
  exit 1
fi

echo "Raw input  : ${RAW}"
echo "Tag        : ${TAG}"
echo "Data npy   : ${DATA_NPY}"
echo "Work dir   : ${WORK}"
echo "Eval dir   : ${EVAL_OUT}"

# --- Stage 1: build leading-30 dataset (log pt_rel by default) -------------
if [[ "${BUFF_SKIP_BUILD:-0}" != "1" ]]; then
  echo
  echo ">>> [1/3] Building leading-30 particle dataset"
  mkdir -p "${DATA_DIR}"
  run_py -m BUFF.scripts.build_jetnet_particle_kin \
    --raw "${RAW}" \
    --out "${DATA_NPY}" \
    --n-leading 30
else
  echo ">>> [1/3] Build SKIPPED"
  if [[ ! -f "${DATA_NPY}" ]]; then
    echo "ERROR: BUFF_SKIP_BUILD=1 but ${DATA_NPY} does not exist" >&2
    exit 1
  fi
fi

# --- Stage 2: train flowBDT and sample -------------------------------------
if [[ "${BUFF_SKIP_TRAIN:-0}" != "1" ]]; then
  echo
  echo ">>> [2/3] Train flowBDT and sample"
  mkdir -p "${WORK}"
  EXTRA="${BUFF_30X3_TRAIN_ARGS:---n-timesteps 30 --duplicate-k 20 --multi-output --flow-type icfm --sigma 0.0 --solver dopri5 --solver-steps 30}"
  # n_jobs=8 across 128 cores -> XGBoost gets 16 threads per fit, 8 fits in parallel.
  # shellcheck disable=SC2086
  run_py -m BUFF.runner.train_and_sample \
    --data "${DATA_NPY}" \
    --output-dir "${WORK}" \
    --n-jobs "${BUFF_30X3_NJOBS:-8}" \
    --n-threads "${BUFF_30X3_NTHREADS:-16}" \
    ${EXTRA}
else
  echo ">>> [2/3] Train+sample SKIPPED"
  if [[ ! -f "${GEN_NPY}" ]]; then
    echo "ERROR: BUFF_SKIP_TRAIN=1 but ${GEN_NPY} does not exist" >&2
    exit 1
  fi
fi

# --- Stage 3: W1 evaluation with bootstrap + truth baseline ---------------
if [[ "${BUFF_SKIP_EVAL:-0}" != "1" ]]; then
  echo
  echo ">>> [3/3] W1 evaluation (bootstrap n=100)"
  mkdir -p "${EVAL_OUT}"
  run_py -m BUFF.scripts.run_jetnet_30x3_eval \
    --real-hdf5 "${RAW}" \
    --gen-npy "${GEN_NPY}" \
    --gen-log-pt \
    --out-dir "${EVAL_OUT}" \
    --n-bootstrap "${BUFF_30X3_NBOOT:-100}" \
    --per-particle-subsample "${BUFF_30X3_PP_SUB:-200000}"
else
  echo ">>> [3/3] Eval SKIPPED"
fi

echo
echo "PIPELINE DONE $(date -Is)"
echo "  Data npy : ${DATA_NPY}"
echo "  Generated: ${GEN_NPY}"
echo "  Eval out : ${EVAL_OUT}"
echo
echo "To view results:"
echo "  cat ${EVAL_OUT}/results.json"
echo "  cat ${EVAL_OUT}/table.tex"
