# MLP Discriminator Results

Classifier test: train an MLP to distinguish real data from generated flowBDT data. Report AUC with bootstrap uncertainty.

## Method

1. Combine real and generated samples with labels (1 = real, 0 = generated)
2. Train/test split (70/30, stratified)
3. StandardScaler normalization
4. Train MLP classifier (2-layer, 128-64 hidden units, ReLU, early stopping)
5. Compute AUC on held-out test set
6. Bootstrap (n=20) for uncertainty

An AUC close to 0.5 indicates the generator is indistinguishable from real data. AUC close to 1.0 means the discriminator easily separates them.

Implementation: `BUFF/evaluation/discriminator.py`

## Results

### JetNet High-Level (12 features, top-quark jets)

| Metric | Value |
|--------|-------|
| AUC (single run, test) | 0.9995 |
| AUC (single run, train) | 0.9999 |
| AUC (bootstrap, n=20) | 0.9994 +/- 0.0002 |

**Interpretation**: The high AUC indicates that while individual feature marginals are well-reproduced (W1 distances are small, see `bootstrap_results.md`), the inter-feature correlations are not fully captured. This is expected for the per-feature regression approach used by flowBDT, where each feature is regressed independently at each timestep. The flow matching ODE couples the features through the shared input at each step, but cannot enforce exact functional relationships (e.g., tau21 = tau2/tau1). This represents an area for future improvement.

### CaloChallenge Dataset 1 (Photons, HLV representation)

| Metric | Value |
|--------|-------|
| AUC (single run) | 0.9996 |
| AUC (bootstrap, n=20) | 0.9995 +/- 0.0002 |
| ROC curve | see `rebuttal/results/calo/roc_curve.pdf` |

Note: CaloChallenge results are on a simplified 8-feature HLV representation, not the full 368-voxel showers. The full voxel-level discriminator test requires the complete preprocessing and generation pipeline.

## How to Reproduce

```bash
# JetNet discriminator
uv run python -c "
from BUFF.evaluation.discriminator import train_discriminator, discriminator_auc_with_errors
import numpy as np
real = np.load('data/jetnet/t_hlv12.npy')
gen = np.load('results/jetnet_highlevel/generated_samples.npy')
print(train_discriminator(real, gen))
print(discriminator_auc_with_errors(real, gen, n_bootstrap=20))
"

# CaloChallenge HLV discriminator
uv run python scripts/run_calo_hlv_eval.py
```
