#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_jetnet_30x3_hicap
#SBATCH --output=slurm_jetnet_30x3_hicap_%j.out
#SBATCH --error=slurm_jetnet_30x3_hicap_%j.err
#SBATCH --export=ALL
#
# Higher-capacity 30x3 retrain to push the jet-level W1 metrics closer to the
# truth floor.  Defaults:
#
#     max_depth        = 6        (was 4)
#     n_estimators     = 200      (was 100)
#     eta              = 0.05     (was 0.1)
#     multi-output     = yes
#     n_timesteps      = 30       (unchanged)
#     duplicate_K      = 20       (unchanged)
#     solver           = dopri5, 30 steps
#
# Expected runtime: ~3x the baseline (depth 6 ~1.5x, n_est 200 ~2x), so
# ~19h for 30 timesteps.  Wall-time 24h leaves a 5h margin.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

RAW="${BUFF_JETNET_RAW:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet_raw/t.hdf5}"
TAG="${BUFF_30X3_TAG:-t30x3_hicap_d6n200}"

DATA_DIR="${BUFF_RESULTS_DIR}/jetnet_30x3/data"
WORK="${BUFF_RESULTS_DIR}/jetnet_30x3/${TAG}"
DATA_NPY="${DATA_DIR}/t30x3_p30_kin.npy"
GEN_NPY="${WORK}/generated_samples.npy"
EVAL_OUT="${WORK}/eval"
EVAL_RENORM_OUT="${WORK}/eval_renorm"

if [[ ! -f "${RAW}" ]]; then
  echo "ERROR: raw JetNet hdf5 not found at ${RAW}" >&2
  exit 1
fi

echo "Raw input  : ${RAW}"
echo "Tag        : ${TAG}"
echo "Data npy   : ${DATA_NPY}  (reused if it exists)"
echo "Work dir   : ${WORK}"

# --- Stage 1: reuse existing leading-30 dataset (built by the baseline run) -
if [[ ! -f "${DATA_NPY}" ]]; then
  echo
  echo ">>> [1/3] Building leading-30 particle dataset (one-time)"
  mkdir -p "${DATA_DIR}"
  run_py -m BUFF.scripts.build_jetnet_particle_kin \
    --raw "${RAW}" \
    --out "${DATA_NPY}" \
    --n-leading 30
else
  echo ">>> [1/3] Reusing existing dataset at ${DATA_NPY}"
fi

# --- Stage 2: higher-capacity train + sample ------------------------------
if [[ "${BUFF_SKIP_TRAIN:-0}" != "1" ]]; then
  echo
  echo ">>> [2/3] Train flowBDT (depth=6, n_est=200, eta=0.05) and sample"
  mkdir -p "${WORK}"
  EXTRA="${BUFF_30X3_TRAIN_ARGS:---n-timesteps 30 --duplicate-k 20 --multi-output --flow-type icfm --sigma 0.0 --solver dopri5 --solver-steps 30 --max-depth 6 --n-estimators 200 --eta 0.05}"
  # shellcheck disable=SC2086
  run_py -m BUFF.runner.train_and_sample \
    --data "${DATA_NPY}" \
    --output-dir "${WORK}" \
    --n-jobs "${BUFF_30X3_NJOBS:-8}" \
    --n-threads "${BUFF_30X3_NTHREADS:-16}" \
    ${EXTRA}
else
  echo ">>> [2/3] Train+sample SKIPPED"
fi

# --- Stage 3: evaluate (raw and with post-hoc pt-renormalization) ---------
if [[ "${BUFF_SKIP_EVAL:-0}" != "1" ]]; then
  echo
  echo ">>> [3/3] W1 evaluation (raw)"
  mkdir -p "${EVAL_OUT}"
  run_py -m BUFF.scripts.run_jetnet_30x3_eval \
    --real-hdf5 "${RAW}" \
    --gen-npy "${GEN_NPY}" \
    --gen-log-pt \
    --out-dir "${EVAL_OUT}" \
    --n-bootstrap "${BUFF_30X3_NBOOT:-100}"

  echo
  echo ">>> [3b/3] W1 evaluation (post-hoc pt-renormalized)"
  mkdir -p "${EVAL_RENORM_OUT}"
  run_py -m BUFF.scripts.run_jetnet_30x3_eval \
    --real-hdf5 "${RAW}" \
    --gen-npy "${GEN_NPY}" \
    --gen-log-pt \
    --renormalize-pt \
    --out-dir "${EVAL_RENORM_OUT}" \
    --n-bootstrap "${BUFF_30X3_NBOOT:-100}"
else
  echo ">>> [3/3] Eval SKIPPED"
fi

echo
echo "HICAP PIPELINE DONE $(date -Is)"
echo "  Generated  : ${GEN_NPY}"
echo "  Eval raw   : ${EVAL_OUT}/results.json"
echo "  Eval renorm: ${EVAL_RENORM_OUT}/results.json"
