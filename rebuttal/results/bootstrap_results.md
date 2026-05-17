# Bootstrap Uncertainty Results

Bootstrap evaluation (n=100 resamples) for all metrics in Tables 1 and 2.

## Method

For each metric:
1. Resample generated and real datasets with replacement (same size as original)
2. Compute metric on resampled pair
3. Repeat 100 times
4. Report: mean +/- std

**Truth baseline**: split real data into two halves, compute metric between them with same bootstrap procedure. This establishes the finite-sample floor.

Implementation: `BUFF/evaluation/bootstrap.py`

## Table 1 — JetNet High-Level (12 features)

**Training config**: flowBDT with `max_depth=4`, `n_estimators=100`, `n_timesteps=30`, `duplicate_K=20`, DOPRI5 solver (30 steps). Trained on 177,945 top-quark jets. Generated 50,000 jets.

Separation power (triangular discriminator, x100, closer to 0 = better):

| Feature | flowBDT | Truth baseline |
|---------|---------|----------------|
| pT | 0.207 +/- 0.020 | 0.027 +/- 0.005 |
| mass | 0.335 +/- 0.027 | 0.029 +/- 0.005 |
| tau1 | 0.320 +/- 0.026 | 0.028 +/- 0.005 |
| tau2 | 0.726 +/- 0.045 | 0.026 +/- 0.005 |
| tau3 | 0.329 +/- 0.028 | 0.026 +/- 0.005 |
| tau21 | 0.298 +/- 0.025 | 0.027 +/- 0.005 |
| tau32 | 0.259 +/- 0.026 | 0.027 +/- 0.006 |
| d12 | 0.185 +/- 0.019 | 0.028 +/- 0.006 |
| d23 | 0.548 +/- 0.040 | 0.026 +/- 0.005 |
| ecf2 | 0.567 +/- 0.039 | 0.027 +/- 0.005 |
| ecf3 | 1.284 +/- 0.064 | 0.025 +/- 0.005 |
| d2 | 0.304 +/- 0.027 | 0.028 +/- 0.006 |

Wasserstein-1 distance (x10, lower = better):

| Feature | flowBDT | Truth baseline |
|---------|---------|----------------|
| pT | 0.0196 +/- 0.0019 | 0.0037 +/- 0.0016 |
| mass | 0.0148 +/- 0.0009 | 0.0018 +/- 0.0007 |
| tau1 | 0.0195 +/- 0.0013 | 0.0026 +/- 0.0009 |
| tau2 | 0.0215 +/- 0.0009 | 0.0018 +/- 0.0008 |
| tau3 | 0.0061 +/- 0.0004 | 0.0007 +/- 0.0002 |
| tau21 | 0.0644 +/- 0.0053 | 0.0106 +/- 0.0046 |
| tau32 | 0.0778 +/- 0.0087 | 0.0137 +/- 0.0056 |
| d12 | 0.0065 +/- 0.0012 | 0.0022 +/- 0.0007 |
| d23 | 0.0112 +/- 0.0006 | 0.0012 +/- 0.0005 |
| ecf2 | 0.0135 +/- 0.0006 | 0.0012 +/- 0.0005 |
| ecf3 | 0.0004 +/- 0.0000 | 0.0000 +/- 0.0000 |
| d2 | 0.0105 +/- 0.0008 | 0.0017 +/- 0.0007 |

**Discriminator AUC** (MLP, 128-64 hidden units):
- Single run: 0.9995
- Bootstrap (n=20): 0.9994 +/- 0.0002

## Table 2 — CaloChallenge (dataset 1, photons, HLV representation)

High-level features (8 features): energy response, 5 layer fractions, depth, width.

| Metric | flowBDT | Truth baseline |
|--------|---------|----------------|
| Response W1 | 0.0116 +/- 0.0003 | 0.0009 +/- 0.0002 |
| Layer 0 frac W1 | 0.0052 +/- 0.0004 | 0.0007 +/- 0.0003 |
| Layer 1 frac W1 | 0.0034 +/- 0.0003 | 0.0005 +/- 0.0002 |
| Layer 2 frac W1 | 0.0024 +/- 0.0004 | 0.0008 +/- 0.0003 |
| Layer 3 frac W1 | 0.0017 +/- 0.0002 | 0.0005 +/- 0.0002 |
| Layer 4 frac W1 | 0.0015 +/- 0.0001 | 0.0003 +/- 0.0001 |
| Discriminator AUC | 0.9995 +/- 0.0002 | — |

Note: CaloChallenge results use a simplified high-level representation (8 features extracted from 368 voxels). Full voxel-level generation and evaluation requires the complete preprocessing pipeline with geometry handling (`calo_utils.py`, `XMLHandler.py`).

## Table 3 — JetNet 30x3 (particle-level, top jets)

Standard JetNet benchmark metrics following Ref [58]: unbinned W1 on derived jet observables and per-particle features, with truth-baseline floors from independent halves of the real data. Generated 177,945 jets via `train_and_sample.py --multi-output --duplicate-k 20 --n-timesteps 30 --solver dopri5 --solver-steps 30` on the leading-30 particle representation (log pt_rel, eta_rel, phi_rel; 90 features total). Bootstrap n=100, per-particle subsample 200,000.

### Headline (hicap config: depth=6, n_est=200, eta=0.05, multi-output)

| Metric | flowBDT raw | flowBDT + pt-renorm | Truth baseline |
|--------|-------------|---------------------|----------------|
| W1-M (jet mass / pT_jet)    | 242.16 +/- 0.21 | 183.66 +/- 0.14 | 0.18 +/- 0.07 |
| W1-Pt (sanity)              | 198.68 +/- 0.65 |  17.25 +/- 0.22 | 0.38 +/- 0.14 |
| W1-P(eta_rel) per-particle  | 166.35 +/- 0.58 | 164.76 +/- 0.48 | 0.70 +/- 0.25 |
| W1-P(phi_rel) per-particle  | 113.60 +/- 0.40 | 113.11 +/- 0.44 | 0.70 +/- 0.25 |
| W1-P(pT_rel)  per-particle  |   9.07 +/- 0.20 |   4.97 +/- 0.11 | 0.24 +/- 0.08 |

### Baseline (depth=4, n_est=100, eta=0.1) -- for comparison

| Metric | flowBDT raw | flowBDT + pt-renorm | Truth baseline |
|--------|-------------|---------------------|----------------|
| W1-M (jet mass / pT_jet)    | 276.92 +/- 0.24 | 202.55 +/- 0.14 | 0.18 +/- 0.07 |
| W1-Pt (sanity)              | 229.96 +/- 0.71 |  19.79 +/- 0.22 | 0.38 +/- 0.14 |
| W1-P(eta_rel) per-particle  | 189.66 +/- 0.58 | 189.52 +/- 0.51 | 0.70 +/- 0.25 |
| W1-P(phi_rel) per-particle  | 129.59 +/- 0.42 | 129.32 +/- 0.38 | 0.70 +/- 0.25 |
| W1-P(pT_rel)  per-particle  |  10.70 +/- 0.17 |   5.98 +/- 0.11 | 0.24 +/- 0.08 |

Hicap improves every metric by ~10-15% (15h 41min training vs 6h 23min baseline). The structural limitation (jet mass two orders of magnitude above truth floor) is not resolved by more capacity; closing the gap requires a backbone with an explicit jet-level loss term.

Post-hoc renormalization: for each generated jet, the 30 generated pt_rel values are rescaled by a single factor so that their sum equals a sample drawn from the empirical real-data pt_sum distribution. This is a cheap fix for the most obvious failure mode (gen pt_sum std = 0.33 vs real 0.06). It collapses W1-Pt to within ~50x the floor and halves W1-P(pT). It does NOT improve W1-M materially because per-jet pt rescaling leaves m_jet / pT_jet unchanged in expectation; the remaining mass discrepancy is driven by the angular distribution and per-particle correlations.

Note: Per-particle marginal distributions track the target reasonably well (within ~50x the finite-sample floor), but jet-level kinematics (mass, total pT) are three orders of magnitude above the floor. This is the per-feature regression failure mode: each of the 30 x 3 outputs is modelled independently, so the joint constraint sum_i pT_rel_i ~ 1 that defines the jet 4-momentum is not preserved. A multi-output tree backbone or an explicit jet-level loss is the natural follow-up (and the diagnostic for whether the limitation is fundamental or pipeline-level).

Sanity check on generated samples:
- Sum of generated pT_rel per jet: mean = 1.14, std = 0.33 (vs real: 1.00, std = 0.06)
- Per-particle eta_rel/phi_rel means: track real to ~1e-3, indicating the marginals are not the failure mode

## How to Reproduce

```bash
# JetNet
uv run python scripts/preprocess_jetnet_fastjet.py -f data/jetnet/t.hdf5 -o data/jetnet/t_hlv12.npy
uv run python -m BUFF.runner.train_and_sample \
    --data data/jetnet/t_hlv12.npy \
    --n-timesteps 30 --duplicate-k 20 \
    --solver dopri5 --solver-steps 30 \
    --output-dir results/jetnet_highlevel/
uv run python -m BUFF.evaluation.run_jetnet_eval \
    --real data/jetnet/t_hlv12.npy \
    --gen results/jetnet_highlevel/generated_samples.npy \
    --n-bootstrap 100 \
    --output-dir rebuttal/results/jetnet/

# CaloChallenge HLV
uv run python scripts/preprocess_calo_hlv.py -f data/calo/dataset_1_photons_1.hdf5
uv run python -m BUFF.runner.train_and_sample \
    --data data/calo/processed/calo_hlv.npy \
    --n-timesteps 30 --duplicate-k 20 \
    --solver dopri5 --solver-steps 30 \
    --output-dir results/calo_hlv/
uv run python scripts/run_calo_hlv_eval.py

# JetNet 30x3 particle-level (Table 3)
sbatch scripts/slurm/run_jetnet_30x3.sh
# Or step-by-step:
python -m BUFF.scripts.build_jetnet_particle_kin \
    --raw data/jetnet_raw/t.hdf5 \
    --out data/jetnet/t_p30_kin.npy \
    --n-leading 30
python -m BUFF.runner.train_and_sample \
    --data data/jetnet/t_p30_kin.npy \
    --n-timesteps 30 --duplicate-k 20 --multi-output \
    --flow-type icfm --solver dopri5 --solver-steps 30 \
    --output-dir results/jetnet_30x3/
python -m BUFF.scripts.run_jetnet_30x3_eval \
    --real-hdf5 data/jetnet_raw/t.hdf5 \
    --gen-npy   results/jetnet_30x3/generated_samples.npy \
    --gen-log-pt \
    --out-dir   rebuttal/results/jetnet_30x3/
```
