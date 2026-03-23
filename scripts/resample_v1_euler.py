"""Re-sample v1 models with Euler solver and re-run full evaluation."""
import pickle
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from BUFF.runner.ode_example import euler_solve
from BUFF.runner.train_and_sample import build_model_fn

# Load saved v1 models
with open("results/jetnet_highlevel/models.pkl", "rb") as f:
    regr = pickle.load(f)
with open("results/jetnet_highlevel/scaler.pkl", "rb") as f:
    meta = pickle.load(f)

scaler = meta["scaler"]
X_min, X_max = meta["X_min"], meta["X_max"]
y_uniques = meta["y_uniques"]
y_probs = meta["y_probs"]
c = meta["c"]
n_t = meta["n_t"]

n_samples = 177945
np.random.seed(1980)

x0 = np.random.normal(size=(n_samples, c))
label_y = y_uniques[np.argmax(
    np.random.multinomial(1, y_probs, size=n_samples), axis=1
)]
mask_y = {label: (label_y == label) for label in y_uniques}
model_fn = build_model_fn(regr, y_uniques, c, n_t, mask_y)

print(f"Sampling {n_samples} events with Euler (N={n_t})...")
t0 = time.time()
solution = euler_solve(x0=x0.reshape(-1), my_model=model_fn, N=n_t)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

solution = solution.reshape(n_samples, c)
solution = scaler.inverse_transform(solution)
solution = np.clip(solution, X_min, X_max)

np.save("results/jetnet_highlevel/generated_samples_euler.npy", solution)
print("Saved to results/jetnet_highlevel/generated_samples_euler.npy")
