#!/bin/bash
# Shared setup for BUFF F2D2 Slurm jobs on NERSC Perlmutter.
#
# Each wrapper script sources this file AFTER its SBATCH directives.
# Override any of the defaults with environment variables at submit time:
#
#   BUFF_REPO=/path/to/BUFF
#   BUFF_CONDA_ENV=/path/to/conda/env   (bin/python must exist)
#   BUFF_USE_UV=1                        (prefer `uv run` over conda)
#   BUFF_RESULTS_DIR=$PSCRATCH/buff_f2d2
#   BUFF_OMP_THREADS=128                 (OpenMP intra-op threads)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — edit here once, or override via the environment
# ---------------------------------------------------------------------------
_DEFAULT_REPO=/pscratch/sd/s/sqian/BUFF
_DEFAULT_CONDA_ENV=/pscratch/sd/s/sqian/BUFF/.venv
_DEFAULT_RESULTS_DIR="$PSCRATCH/buff_f2d2"
# ---------------------------------------------------------------------------

# Resolve repo: prefer env, then SLURM_SUBMIT_DIR if it looks like BUFF, then default.
if [[ -z "${BUFF_REPO:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    BUFF_REPO="${SLURM_SUBMIT_DIR}"
  else
    BUFF_REPO="${_DEFAULT_REPO}"
  fi
fi

if [[ ! -d "${BUFF_REPO}" ]]; then
  echo "ERROR: BUFF_REPO=${BUFF_REPO} does not exist" >&2
  exit 1
fi

# BUFF imports itself as `BUFF.*`, so PYTHONPATH must point to the *parent* dir.
BUFF_PARENT_DIR="$(cd "${BUFF_REPO}/.." && pwd)"

export BUFF_REPO
export BUFF_PARENT_DIR
export BUFF_CONDA_ENV="${BUFF_CONDA_ENV:-${_DEFAULT_CONDA_ENV}}"
export BUFF_RESULTS_DIR="${BUFF_RESULTS_DIR:-${_DEFAULT_RESULTS_DIR}}"
export BUFF_OMP_THREADS="${BUFF_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-128}}"
export BUFF_USE_UV="${BUFF_USE_UV:-0}"

# Threading hygiene for CPU-only jobs.  Set every known var so neither
# XGBoost, numpy/MKL, PyTorch, nor OpenBLAS oversubscribes the node.
export OMP_NUM_THREADS="${BUFF_OMP_THREADS}"
export MKL_NUM_THREADS="${BUFF_OMP_THREADS}"
export OPENBLAS_NUM_THREADS="${BUFF_OMP_THREADS}"
export NUMEXPR_MAX_THREADS="${BUFF_OMP_THREADS}"
export NUMEXPR_NUM_THREADS="${BUFF_OMP_THREADS}"
export PYTORCH_NUM_THREADS="${BUFF_OMP_THREADS}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="${BUFF_PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${BUFF_RESULTS_DIR}"

echo "========== BUFF F2D2 Slurm job =========="
echo "JOB       : ${SLURM_JOB_NAME:-<none>}  (id ${SLURM_JOB_ID:-local})"
echo "NODE      : $(hostname)"
echo "CPUS/TASK : ${SLURM_CPUS_PER_TASK:-?}  (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
echo "REPO      : ${BUFF_REPO}"
echo "RESULTS   : ${BUFF_RESULTS_DIR}"
echo "START     : $(date -Is)"
echo "========================================="

cd "${BUFF_REPO}"

# Pick interpreter: uv if requested and available, otherwise conda env.
if [[ "${BUFF_USE_UV}" == "1" ]] && command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python)
  echo "Interpreter: uv run python"
else
  if [[ ! -x "${BUFF_CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: no ${BUFF_CONDA_ENV}/bin/python.  Either create the env, or set BUFF_USE_UV=1." >&2
    exit 1
  fi
  module load conda 2>/dev/null || true
  eval "$(conda shell.bash hook)"
  conda activate "${BUFF_CONDA_ENV}"
  PY_RUN=(python)
  echo "Interpreter: ${BUFF_CONDA_ENV}/bin/python"
fi

# Import BUFF as a package — run scripts from BUFF_PARENT_DIR so `python -m BUFF.*` works.
cd "${BUFF_PARENT_DIR}"

run_py() {
  echo "+ ${PY_RUN[*]} $*"
  "${PY_RUN[@]}" "$@"
}
