"""Test different numbers of Euler steps with the same trained models."""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from BUFF.runner.ode_example import euler_solve
from BUFF.runner.train_and_sample import build_model_fn
from BUFF.evaluation.discriminator import train_discriminator

# Load saved models
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
real = np.load("data/jetnet/t_hlv12.npy")

n_samples = 177945

for N_steps in [30, 60, 100, 200]:
    np.random.seed(1980)
    x0 = np.random.normal(size=(n_samples, c))
    label_y = y_uniques[np.argmax(
        np.random.multinomial(1, y_probs, size=n_samples), axis=1
    )]
    mask_y = {label: (label_y == label) for label in y_uniques}
    model_fn = build_model_fn(regr, y_uniques, c, n_t, mask_y)

    t0 = time.time()
    solution = euler_solve(x0=x0.reshape(-1), my_model=model_fn, N=N_steps)
    elapsed = time.time() - t0
    solution = solution.reshape(n_samples, c)
    solution = scaler.inverse_transform(solution)
    solution = np.clip(solution, X_min, X_max)

    r = train_discriminator(real, solution)
    corr_real = np.corrcoef(real.T)
    corr_gen = np.corrcoef(solution.T)
    diff = np.abs(corr_real - corr_gen)
    np.fill_diagonal(diff, 0)
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1)

    print(f"Euler N={N_steps:3d}: AUC={r['auc_test']:.4f}  mean|Δρ|={diff[mask].mean():.4f}  max|Δρ|={diff[mask].max():.4f}  ({elapsed:.1f}s)")
