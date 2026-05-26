# Drop-cointegrated test — internal notes (not in submission)

Diagnostic experiment run after the cointegration finding (top pair-AUC =
`d_{23} x d_2 = 0.85` on hicap+strip, `0.79` on xgb+residual).  Not added
to the manuscript or reply: held in reserve for follow-up rounds or the
next paper, in case the referee pushes further on the discriminator-AUC
ceiling.

## Setup

Two complementary checks, both run on the same JetNet HLV setup as the
main rebuttal:

- **Post-hoc**: take existing generated samples (baseline / hicap+strip /
  xgb-residual), drop `d_2` (col 11) and/or `d_{23}` (col 1) from both
  real and gen, and re-train the MLP discriminator on the reduced
  feature set.  Single-shot AUCs (no bootstrap).
- **Retrain**: train BUFF on 11 features (drop the derived `d_2`
  observable from training), sample, evaluate AUC on the 11-feature
  representation.  Hyper-parameters: hicap config (multi-output XGB,
  depth=6, n_est=200, eta=0.05, K=20, n_t=30, dopri5 30 steps).

## Results

### Post-hoc

| sample | all 12 | drop d_2 | drop d_{23} | drop both |
|---|---|---|---|---|
| baseline       | 0.9982 | 0.9965 | **0.9731** | 0.9731 |
| hicap+strip    | 0.9977 | 0.9954 | **0.9730** | 0.9735 |
| xgb+residual   | 0.9906 | 0.9869 | **0.9194** | 0.9189 |

Reading:
- Dropping `d_2` alone barely changes AUC (delta -0.002 to -0.004).
- Dropping `d_{23}` alone removes 2.5--7% of AUC, the dominant signal.
- Dropping both ~= dropping `d_{23}` alone (the pair is genuinely
  cointegrated, carrying the same information).
- The drop is biggest for xgb+residual (0.9906 -> 0.9194) because the
  residual two-pass had already flattened the *other* joint
  correlations, so removing `d_{23}` reveals more cleanly that the
  remaining signal lives in this pair.

### Retrain on 11 features (dropped `d_2`)

`auc_bootstrap.json` (bootstrap n=20):

    AUC = 0.9998 +/- 0.0001   (vs 0.9974 for the 12-feature hicap+strip)
    boundary_fraction = 13.4%  (vs ~3% in the 12-feature run)
    per-feature AUC : 0.52-0.59

So **retraining on 11 features (no d_2) made things slightly worse**.
Consistent with the post-hoc finding that `d_2` was *not* the
load-bearing feature.  Without `d_2` as a redundant cross-check, the
per-feature regression had slightly worse marginal fidelity (per-feature
AUCs went up a touch, boundary clipping went up substantially), so AUC
got a hair worse.

## Why this isn't in the rebuttal

The pair-AUC localisation in the main reply already establishes the
cointegration point qualitatively; this is a quantitative confirmation
but with a refined interpretation (d_{23} is load-bearing, not d_2)
that needs more careful presentation than a quick paragraph allows.

## What the clean follow-up would be

A retrain *without* `d_{23}` -- but `d_{23}` is the independent
kt-splitting scale, not algebraically recoverable from the other 11
features.  Doing this would require defining a different evaluation
target (11-feature reduced representation instead of 12-feature
standard).  Save for the next paper.

## Reproduction

- Slurm:   `scripts/slurm/run_drop_cointegrated.sh`
- Job ID:  53429720 (completed 2026-05-26, 1h47m)
- Working dir: `$PSCRATCH/buff_f2d2/jetnet_drop_cointegrated/`
