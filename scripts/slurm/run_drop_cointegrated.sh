#!/bin/bash
#SBATCH --account=m2612
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=buff_drop_cointegrated
#SBATCH --output=slurm_drop_cointegrated_%j.out
#SBATCH --error=slurm_drop_cointegrated_%j.err
#SBATCH --export=ALL
#
# Test the cointegration hypothesis empirically.
#
# Diagnosis says (d_{23}, d_2) is the dominant pair driving discriminator
# AUC.  Two complementary checks:
#
#   (A) POST-HOC subset:  on existing 12-feature samples (baseline,
#       hicap+strip, xgb+residual), drop d_2 (col 11), d_{23} (col 1),
#       or both, and re-train the MLP discriminator on the reduced
#       feature set.  Quick sanity check that bounds how much of the
#       AUC signal is *carried by* those features.
#
#   (B) RETRAIN-WITHOUT-D2:  drop d_2 (the derivable D_2 observable
#       defined as ecf_3 * sum(pt) / ecf_2^2) from training data and
#       retrain BUFF on 11 features.  Sample and evaluate AUC on the
#       new 11-feature representation.  This is the cleaner test --
#       if the cointegration was the dominant signal, the AUC should
#       drop significantly.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/scripts/slurm/_common.sh"

REAL12="${BUFF_REAL12:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv12.npy}"
REAL11="${BUFF_REAL11:-/pscratch/sd/s/sqian/buff_f2d2/data/jetnet/t_hlv11_no_d2.npy}"
BASELINE_GEN="${BUFF_BASELINE_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_t_hlv12/generated_samples.npy}"
HICAP_GEN="${BUFF_HICAP_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_hlv/jetnet_hlv_hicap_d6n200_strip/generated_samples.npy}"
RESIDUAL_GEN="${BUFF_RESIDUAL_GEN:-/pscratch/sd/s/sqian/buff_f2d2/jetnet_hlv/jetnet_hlv_xgb_residual_d6n200/generated_samples.npy}"

ROOT="${BUFF_RESULTS_DIR}/jetnet_drop_cointegrated"
mkdir -p "${ROOT}"

# --- Stage 0: build 11-feature data (drop d_2 col 11) ---------------------
if [[ ! -f "${REAL11}" ]]; then
  echo
  echo ">>> [0] Build 11-feature data (drop col 11 = d_2_obs)"
  run_py -c "
import numpy as np
x = np.load('${REAL12}')
assert x.shape[1] == 12, x.shape
y = np.delete(x, 11, axis=1)  # drop d_2_obs (col 11)
np.save('${REAL11}', y.astype(np.float32))
print(f'saved ${REAL11}: shape={y.shape}, dtype={y.dtype}')
"
fi

# --- Stage 1: POST-HOC drop-feature AUC on existing samples ---------------
echo
echo ">>> [1] POST-HOC drop-feature AUC on existing 12-feature samples"
run_py -c "
import numpy as np, json, os
from BUFF.evaluation.discriminator import train_discriminator

real = np.load('${REAL12}')
gens = {
    'baseline':     np.load('${BASELINE_GEN}'),
    'hicap_strip':  np.load('${HICAP_GEN}'),
    'xgb_residual': np.load('${RESIDUAL_GEN}'),
}
features = ['d12','d23','mass','pt','tau1','tau2','tau3','tau21','tau32','ecf2','ecf3','d2_obs']

out = {}
for name, gen in gens.items():
    out[name] = {}
    print(f'--- {name} ---')
    for label, drop in [('12_all', []), ('11_no_d2_obs', [11]), ('11_no_d23', [1]), ('10_no_both', [1, 11])]:
        keep = [i for i in range(12) if i not in drop]
        r, g = real[:, keep], gen[:, keep]
        n = min(len(r), len(g))
        res = train_discriminator(r[:n], g[:n], seed=42)
        out[name][label] = {'auc_test': float(res['auc_test']), 'auc_train': float(res['auc_train']), 'features_kept': [features[i] for i in keep]}
        print(f'  {label:20s}  AUC_test = {res[\"auc_test\"]:.4f}  AUC_train = {res[\"auc_train\"]:.4f}')

os.makedirs('${ROOT}', exist_ok=True)
with open('${ROOT}/posthoc_drop_auc.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'saved ${ROOT}/posthoc_drop_auc.json')
"

# --- Stage 2: RETRAIN BUFF on 11 features (drop d_2_obs) ------------------
WORK="${ROOT}/retrain_11feat_drop_d2"
GEN11="${WORK}/generated_samples.npy"
mkdir -p "${WORK}"

if [[ "${BUFF_SKIP_RETRAIN:-0}" != "1" ]]; then
  echo
  echo ">>> [2] RETRAIN BUFF on 11 features (drop d_2_obs)"
  run_py -m BUFF.runner.train_and_sample \
    --data "${REAL11}" \
    --output-dir "${WORK}" \
    --n-timesteps 30 --duplicate-k 20 \
    --multi-output \
    --flow-type icfm --sigma 0.0 \
    --solver dopri5 --solver-steps 30 \
    --max-depth 6 --n-estimators 200 --eta 0.05 \
    --n-jobs 1 --n-threads "${OMP_NUM_THREADS}"
fi

# --- Stage 3: AUC + per-feature on the retrained 11-feature samples -------
echo
echo ">>> [3] AUC + per-feature breakdown (retrained, 11-feature)"
run_py -m BUFF.scripts.run_jetnet_hlv_auc \
  --real "${REAL11}" \
  --gen "${GEN11}" \
  --out-dir "${WORK}/eval" \
  --n-bootstrap 20

# --- Stage 4: pair-AUC (11x11 heatmap) ------------------------------------
echo
echo ">>> [4] pair-AUC on retrained 11-feature samples"
run_py -m BUFF.scripts.run_jetnet_hlv_pair_auc \
  --real "${REAL11}" \
  --gen "${GEN11}" \
  --out-dir "${WORK}/pairs" \
  --n-events 30000

echo
echo "DROP-COINTEGRATED PIPELINE DONE $(date -Is)"
echo "  Post-hoc JSON : ${ROOT}/posthoc_drop_auc.json"
echo "  Retrain dir   : ${WORK}/"
echo "  Eval AUC json : ${WORK}/eval/auc_bootstrap.json"
