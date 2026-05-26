#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_fpd_kpd
#SBATCH --output=slurm_fpd_kpd_%j.out
#SBATCH --error=slurm_fpd_kpd_%j.err
#SBATCH --export=ALL
#
# Compute FPD and KPD (and FPND if torch_geometric is installable) on all
# JetNet flowBDT configurations we care about, for the rebuttal.
#
#   - HLV baseline  (12 features, depth=4, n_est=100, no multi-output)
#   - HLV hicap+strip (12 derived, depth=6, n_est=200, multi-output, strip-derived)
#   - 30x3 baseline   (depth=4, n_est=100, multi-output)
#   - 30x3 hicap      (depth=6, n_est=200, multi-output)
#
# Each run also computes a real-vs-real truth-baseline floor.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

RAW="${BUFF_JETNET_RAW:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet_raw/t.hdf5}"
HLV_REAL="${BUFF_HLV_REAL:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
HLV_GEN_BASE="${BUFF_HLV_GEN_BASE:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_t_hlv12/generated_samples.npy}"
HLV_GEN_HICAP="${BUFF_HLV_GEN_HICAP:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_hlv/jetnet_hlv_hicap_d6n200_strip/generated_samples.npy}"
P30_GEN_BASE="${BUFF_P30_GEN_BASE:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_30x3/t30x3/generated_samples.npy}"
P30_GEN_HICAP="${BUFF_P30_GEN_HICAP:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_30x3/t30x3_hicap_d6n200/generated_samples.npy}"

OUT_BASE="${BUFF_RESULTS_DIR}/fpd_kpd"
N_EVENTS="${BUFF_FPD_NEVENTS:-50000}"

mkdir -p "${OUT_BASE}"

# Try to install torch_geometric for FPND (best-effort; FPD/KPD work without)
if ! "${BUFF_CONDA_ENV}/bin/python" -c 'import torch_geometric' 2>/dev/null; then
  echo "Attempting to install torch_geometric for FPND..."
  "${BUFF_CONDA_ENV}/bin/python" -m pip install --user torch_geometric 2>&1 | tail -3 || echo "  (install failed -- FPND will be skipped)"
fi

# --- HLV baseline ---------------------------------------------------------
if [[ -f "${HLV_GEN_BASE}" ]]; then
  echo
  echo ">>> [1/4] HLV baseline"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${HLV_REAL}" --gen "${HLV_GEN_BASE}" \
    --mode hlv --jet-type t \
    --out-dir "${OUT_BASE}/hlv_baseline" --n-events "${N_EVENTS}"
else
  echo "Skipping HLV baseline (no gen file at ${HLV_GEN_BASE})"
fi

# --- HLV hicap + strip ----------------------------------------------------
if [[ -f "${HLV_GEN_HICAP}" ]]; then
  echo
  echo ">>> [2/4] HLV hicap+strip"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${HLV_REAL}" --gen "${HLV_GEN_HICAP}" \
    --mode hlv --jet-type t \
    --out-dir "${OUT_BASE}/hlv_hicap_strip" --n-events "${N_EVENTS}"
else
  echo "Skipping HLV hicap (no gen file at ${HLV_GEN_HICAP})"
fi

# --- 30x3 baseline --------------------------------------------------------
if [[ -f "${P30_GEN_BASE}" ]]; then
  echo
  echo ">>> [3/4] 30x3 baseline (log-pt -> raw-pt for gen)"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${RAW}" --gen "${P30_GEN_BASE}" --gen-log-pt \
    --mode particle --jet-type t \
    --out-dir "${OUT_BASE}/p30_baseline" --n-events "${N_EVENTS}"
else
  echo "Skipping 30x3 baseline"
fi

# --- 30x3 hicap -----------------------------------------------------------
if [[ -f "${P30_GEN_HICAP}" ]]; then
  echo
  echo ">>> [4/4] 30x3 hicap"
  run_py -m BUFF.scripts.run_jetnet_hlv_fpd \
    --real "${RAW}" --gen "${P30_GEN_HICAP}" --gen-log-pt \
    --mode particle --jet-type t \
    --out-dir "${OUT_BASE}/p30_hicap" --n-events "${N_EVENTS}"
else
  echo "Skipping 30x3 hicap"
fi

echo
echo "FPD/KPD PIPELINE DONE $(date -Is)"
echo "Outputs under: ${OUT_BASE}"
