# Consistency Check Results — tau21 / tau32

Compare directly generated substructure ratios against ratios computed from independently generated components. A small difference confirms the model captures inter-variable correlations correctly.

## Method

For the JetNet 12-feature setup, flowBDT independently generates all 12 features including:
- tau1, tau2, tau3 (N-subjettiness)
- tau21 = tau2/tau1 (generated directly)
- tau32 = tau3/tau2 (generated directly)

**Consistency test**: compare
- Generated tau21 vs (generated tau2) / (generated tau1)
- Generated tau32 vs (generated tau3) / (generated tau2)

using Wasserstein-1 distance.

Implementation: `BUFF/evaluation/consistency.py`

## Results

### tau21 Consistency

| Comparison | W1 distance |
|------------|-------------|
| generated tau21 vs tau2/tau1 (from gen) | 0.0164 |
| real tau21 vs tau2/tau1 (from real) | 0.0000 (exact by construction) |

### tau32 Consistency

| Comparison | W1 distance |
|------------|-------------|
| generated tau32 vs tau3/tau2 (from gen) | 0.0295 |
| real tau32 vs tau3/tau2 (from real) | 0.0000 (exact by construction) |

### Consistency plots

- `rebuttal/results/jetnet/consistency_tau21.pdf` — Overlay of generated tau21 vs derived tau2/tau1
- `rebuttal/results/jetnet/consistency_tau32.pdf` — Overlay of generated tau32 vs derived tau3/tau2

## Interpretation

- The real data has W1 = 0 because tau21 is computed as tau2/tau1, so the "direct" and "derived" values are identical.
- The flowBDT generates each feature independently via separate per-feature regressors, so the consistency check measures whether the model has learned the inter-feature correlations.
- W1 = 0.016 for tau21 and 0.030 for tau32 are small values compared to the feature ranges (tau21 ~ [0, 1.3], tau32 ~ [0.1, 1.3]), indicating the model has captured the key correlations.
- This is a strong test because perfect consistency would require the flow model to learn the exact functional relationship tau21 = tau2/tau1 from data alone.

## How to Reproduce

```bash
uv run python -m BUFF.evaluation.run_jetnet_eval \
    --real data/jetnet/t_hlv12.npy \
    --gen results/jetnet_highlevel/generated_samples.npy \
    --n-bootstrap 100 \
    --output-dir rebuttal/results/jetnet/
```
