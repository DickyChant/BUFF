"""Re-sample from saved v3 models with Euler solver instead of DOPRI5.

Tests whether DOPRI5 intermediate evaluations (which blend adjacent
timestep models) contribute to correlation errors.
"""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from BUFF.runner.ode_example import euler_solve, midpoint_solve, dpori5_solve_numpy
from BUFF.runner.train_and_sample import build_model_fn

# Load saved models and metadata
with open("results/jetnet_highlevel_v3/models.pkl", "rb") as f:
    regr = pickle.load(f)
with open("results/jetnet_highlevel_v3/scaler.pkl", "rb") as f:
    meta = pickle.load(f)

scaler = meta["scaler"]
X_min, X_max = meta["X_min"], meta["X_max"]
y_uniques = meta["y_uniques"]
y_probs = meta["y_probs"]
c = meta["c"]
n_t = meta["n_t"]

n_samples = 177945  # same as training

np.random.seed(1980)

x0 = np.random.normal(size=(n_samples, c))
label_y = y_uniques[np.argmax(
    np.random.multinomial(1, y_probs, size=n_samples), axis=1
)]
mask_y = {label: (label_y == label) for label in y_uniques}

model_fn = build_model_fn(regr, y_uniques, c, n_t, mask_y)

for solver_name, solver_fn in [("euler", euler_solve), ("midpoint", midpoint_solve)]:
    np.random.seed(1980)
    x0 = np.random.normal(size=(n_samples, c))

    print(f"\n=== {solver_name} solver (N={n_t}) ===")
    t0 = time.time()
    solution = solver_fn(x0=x0.reshape(-1), my_model=model_fn, N=n_t)
    elapsed = time.time() - t0
    print(f"Sampling: {elapsed:.1f}s")

    solution = solution.reshape(n_samples, c)
    solution = scaler.inverse_transform(solution)
    solution = np.clip(solution, X_min, X_max)

    np.save(f"results/jetnet_highlevel_v3/generated_{solver_name}.npy", solution)

    # Quick discriminator test
    from BUFF.evaluation.discriminator import train_discriminator
    real = np.load("data/jetnet/t_hlv12.npy")
    r = train_discriminator(real, solution)
    print(f"Discriminator AUC: {r['auc_test']:.4f}")

    # Correlation error
    corr_real = np.corrcoef(real.T)
    corr_gen = np.corrcoef(solution.T)
    diff = np.abs(corr_real - corr_gen)
    np.fill_diagonal(diff, 0)
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
    print(f"Mean |Δρ|: {diff[mask].mean():.4f}, Max |Δρ|: {diff[mask].max():.4f}")
