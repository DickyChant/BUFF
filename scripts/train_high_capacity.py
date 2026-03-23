"""Train with higher capacity models and test with Euler solver.

Test: depth=6, n_estimators=200 on 9 independent features.
This should reduce the per-evaluation velocity prediction error,
which is the dominant source of correlation degradation.
"""
import sys
import time

import numpy as np
import pickle

sys.path.insert(0, ".")
from BUFF.runner.train_and_sample import (
    load_data, prepare_scaling, train_batched, build_model_fn,
    strip_derived_features, restore_derived_features,
    JETNET_INDEPENDENT_INDICES,
)
from BUFF.runner.ode_example import euler_solve
from BUFF.evaluation.discriminator import train_discriminator
from sklearn.preprocessing import MinMaxScaler

import argparse

np.random.seed(1980)
import torch
torch.manual_seed(1980)

# Load data
X_raw, y = load_data("data/jetnet/t_hlv12.npy")
print(f"Loaded {X_raw.shape}")

# Strip derived features
X, derived_info = strip_derived_features(X_raw)
print(f"Stripped to {X.shape[1]} independent features")

X_min = X.min(axis=0)
X_max = X.max(axis=0)
b, c = X.shape

perm = np.random.permutation(b)
X, y = X[perm], y[perm]

# Create mock args for train_batched
class Args:
    n_estimators = 200
    eta = 0.05
    max_depth = 6
    n_threads = 16
    reg_lambda = 0.1
    reg_alpha = 0.2
    subsample = 1.0
    tree_method = "hist"
    multi_output = False  # per-feature is better

args = Args()

n_t = 30
duplicate_K = 20  # Use K=20 for faster training

# Scale
X_scaled, scaler, mask_y, y_uniques, y_probs = prepare_scaling(X, y, duplicate_K)

# Train
print(f"Training: depth={args.max_depth}, n_est={args.n_estimators}, "
      f"eta={args.eta}, n_t={n_t}, K={duplicate_K}")
t0 = time.time()
regr = train_batched(X_scaled, y, duplicate_K, n_t, "icfm", 0.0,
                     mask_y, y_uniques, c, args)
print(f"Training: {time.time()-t0:.0f}s")

# Sample with Euler (N = n_t for optimal quality)
n_samples = b
x0 = np.random.normal(size=(n_samples, c))
label_y = y_uniques[np.argmax(
    np.random.multinomial(1, y_probs, size=n_samples), axis=1
)]
mask_y_fake = {label: (label_y == label) for label in y_uniques}
model_fn = build_model_fn(regr, y_uniques, c, n_t, mask_y_fake)

print("Sampling with Euler...")
solution = euler_solve(x0=x0.reshape(-1), my_model=model_fn, N=n_t)
solution = solution.reshape(n_samples, c)
solution = scaler.inverse_transform(solution)
solution = np.clip(solution, X_min, X_max)

# Restore derived features
solution_full = restore_derived_features(solution, derived_info)
print(f"Restored to {solution_full.shape[1]} features")

# Evaluate
real = np.load("data/jetnet/t_hlv12.npy")

# 9 independent features
r9 = train_discriminator(real[:, JETNET_INDEPENDENT_INDICES],
                         solution_full[:, JETNET_INDEPENDENT_INDICES])
# All 12
r12 = train_discriminator(real, solution_full)

corr_real = np.corrcoef(real.T)
corr_gen = np.corrcoef(solution_full.T)
diff = np.abs(corr_real - corr_gen)
np.fill_diagonal(diff, 0)
mask_tri = np.triu(np.ones_like(diff, dtype=bool), k=1)

print(f"\n=== Results: depth={args.max_depth}, n_est={args.n_estimators}, Euler N={n_t} ===")
print(f"AUC (9 indep): {r9['auc_test']:.4f}")
print(f"AUC (12 feat): {r12['auc_test']:.4f}")
print(f"Mean |Δρ|: {diff[mask_tri].mean():.4f}")
print(f"Max  |Δρ|: {diff[mask_tri].max():.4f}")

# Save for further analysis
np.save("results/jetnet_highlevel_v3/generated_highcap_euler.npy", solution_full)
