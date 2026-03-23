"""
BUFF train-and-sample pipeline.

Clean CLI script extracted from playground.ipynb.  Loads data, trains
per-timestep XGBoost regressors via conditional flow matching, samples
with a configurable ODE solver, and saves generated samples + models.

Improvements over v1:
- Batched training: builds flow pairs one timestep at a time (30x less memory)
- Derived features: strips tau21/tau32/d2_obs before training, derives post-gen
- Multi-output XGBoost: one model per timestep predicting all features jointly
- Cholesky correction: optional post-processing to match real correlation structure
- OT-CFM: optimal transport flow matching for better trajectories

Usage
-----
uv run python -m BUFF.runner.train_and_sample \
    --data path/to/data.h5 \
    --features d12,d2,mass,pt,tau1,tau3 \
    --n-timesteps 30 --duplicate-k 100 \
    --solver dopri5 --solver-steps 30 \
    --output-dir results/jetnet_highlevel/
"""

import argparse
import copy
import os
import pickle
import time
from functools import partial

import h5py
import numpy as np
import torch
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)
from tqdm import tqdm, trange

from BUFF.runner.ode_example import (
    dpori5_solve_numpy,
    euler_solve,
    midpoint_solve,
)


# ── Derived feature definitions ──────────────────────────────────────
# JetNet 12-feature order: d12, d2, mass, pt, tau1, tau2, tau3,
#                           tau21, tau32, ecf2, ecf3, d2_obs
# Indices:                   0    1    2     3    4     5     6
#                            7     8     9     10    11

def _derive_tau21(tau2, tau1):
    return tau2 / np.clip(tau1, 1e-8, None)

def _derive_tau32(tau3, tau2):
    return tau3 / np.clip(tau2, 1e-8, None)

def _derive_d2_obs(ecf3, ecf2):
    return ecf3 / np.clip(ecf2**2, 1e-16, None)

JETNET_DERIVED_FEATURES = {
    # feature_name: (index_in_12, parent_indices, derive_fn)
    "tau21": (7, (5, 4), _derive_tau21),
    "tau32": (8, (6, 5), _derive_tau32),
    "d2_obs": (11, (10, 9), _derive_d2_obs),
}

# Indices to keep for training (the 9 independent features)
JETNET_INDEPENDENT_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10]  # d12,d2,mass,pt,tau1-3,ecf2,ecf3


def strip_derived_features(X):
    """Remove derived features, keeping only independent ones.

    Returns (X_independent, derived_info) where derived_info allows
    reconstruction via restore_derived_features().
    Assumes 12-column JetNet HLV layout.
    """
    if X.shape[1] != 12:
        raise ValueError(f"Expected 12 columns for JetNet HLV, got {X.shape[1]}")
    return X[:, JETNET_INDEPENDENT_INDICES], {
        "original_ncols": 12,
        "independent_indices": JETNET_INDEPENDENT_INDICES,
        "derived": JETNET_DERIVED_FEATURES,
    }


def restore_derived_features(X_indep, derived_info):
    """Reconstruct full 12-feature array from 9 independent features."""
    if derived_info is None:
        return X_indep

    N = X_indep.shape[0]
    X_full = np.zeros((N, derived_info["original_ncols"]))

    # Place independent features
    for new_idx, orig_idx in enumerate(derived_info["independent_indices"]):
        X_full[:, orig_idx] = X_indep[:, new_idx]

    # Derive tau21, tau32, d2_obs from their parents
    for name, (target_idx, parent_indices, fn) in derived_info["derived"].items():
        parents = [X_full[:, pi] for pi in parent_indices]
        X_full[:, target_idx] = fn(*parents)

    return X_full


# ── Cholesky correlation correction ─────────────────────────────────

def cholesky_correction(gen, real):
    """Match the correlation structure of generated data to real data.

    Preserves marginal means/stds while correcting the correlation matrix.
    """
    from scipy.linalg import cholesky, solve_triangular

    gen_mean = gen.mean(axis=0)
    real_mean = real.mean(axis=0)
    gen_std = gen.std(axis=0)
    real_std = real.std(axis=0)

    # Standardize both
    gen_z = (gen - gen_mean) / np.clip(gen_std, 1e-10, None)
    real_z = (real - real_mean) / np.clip(real_std, 1e-10, None)

    # Correlation matrices
    gen_corr = np.corrcoef(gen_z.T)
    real_corr = np.corrcoef(real_z.T)

    # Regularize for numerical stability
    eps = 1e-6
    gen_corr += eps * np.eye(gen_corr.shape[0])
    real_corr += eps * np.eye(real_corr.shape[0])

    L_gen = cholesky(gen_corr, lower=True)
    L_real = cholesky(real_corr, lower=True)

    # Transform: gen_corrected = L_real @ inv(L_gen) @ gen_z
    gen_decorr = solve_triangular(L_gen, gen_z.T, lower=True)
    gen_recorr = L_real @ gen_decorr

    # Restore to original scale (using generated marginals)
    gen_corrected = gen_recorr.T * gen_std + gen_mean

    return gen_corrected


def parse_args():
    p = argparse.ArgumentParser(description="BUFF: train flowBDT and sample")
    p.add_argument("--data", required=True, help="Path to input data (.h5 or .npy)")
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated HDF5 dataset names (for .h5 files)",
    )
    p.add_argument("--n-timesteps", type=int, default=30)
    p.add_argument("--duplicate-k", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=0.1)
    p.add_argument("--reg-alpha", type=float, default=0.2)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--tree-method", type=str, default="hist")
    p.add_argument(
        "--flow-type",
        choices=["icfm", "otcfm", "sbcfm"],
        default="icfm",
        help="Flow matching variant",
    )
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument(
        "--solver", choices=["euler", "midpoint", "dopri5"], default="dopri5"
    )
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=0, help="0 = same as training set")
    p.add_argument("--seed", type=int, default=1980)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for training loop")
    p.add_argument("--n-threads", type=int, default=16, help="CPU threads per XGBoost model")
    p.add_argument("--output-dir", type=str, default="results/")
    p.add_argument(
        "--strip-derived", action="store_true",
        help="Strip derived features (tau21,tau32,d2_obs) and derive post-generation",
    )
    p.add_argument(
        "--multi-output", action="store_true",
        help="Use multi-output XGBoost (one model per timestep, all features jointly)",
    )
    p.add_argument(
        "--cholesky", action="store_true",
        help="Apply Cholesky correlation correction post-generation",
    )
    p.add_argument(
        "--real-data-for-cholesky", type=str, default=None,
        help="Path to real data for Cholesky correction (default: use training data)",
    )
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────

def load_data(path, features=None):
    """Load data from HDF5 or npy file → (X, y) with X shape (N, d)."""
    if path.endswith(".h5") or path.endswith(".hdf5"):
        with h5py.File(path, "r") as f:
            if features is None:
                features = list(f.keys())
            arrays = [f[name][()] for name in features]
        X = np.column_stack(arrays)
    elif path.endswith(".npy"):
        X = np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    y = np.zeros(len(X))
    return X, y


# ── Training ──────────────────────────────────────────────────────────

def build_flow_matcher(flow_type, sigma):
    if flow_type == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif flow_type == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif flow_type == "sbcfm":
        return SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    raise ValueError(flow_type)


def prepare_scaling(X, y, duplicate_K):
    """Scale data and prepare class masks (no flow pair allocation)."""
    b, c = X.shape

    y_uniques, y_counts = np.unique(y, return_counts=True)
    y_probs = y_counts / y_counts.sum()
    mask_y = {}
    for label in y_uniques:
        m = (y == label)
        mask_y[label] = np.tile(m, duplicate_K)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, mask_y, y_uniques, y_probs


def train_batched(X_scaled, y, duplicate_K, n_t, flow_type, sigma, mask_y,
                  y_uniques, c, args):
    """Train models one timestep at a time to avoid OOM.

    Instead of pre-allocating all (n_t, N*K, d) arrays, builds flow pairs
    for a single timestep, trains its regressors, then discards the data.
    Memory: O(N*K*d) instead of O(n_t*N*K*d).
    """
    FM = build_flow_matcher(flow_type, sigma)
    t_levels = np.linspace(1e-3, 1, num=n_t)

    X1 = np.tile(X_scaled, (duplicate_K, 1))

    multi_output = args.multi_output

    if multi_output:
        # regr_mo[class_idx][timestep] = one multi-output model
        regr_mo = [[None for _ in range(n_t)] for _ in y_uniques]
    else:
        # regr[class_idx][timestep][feature] = one single-output model
        regr = [[[None for _ in range(c)] for _ in range(n_t)] for _ in y_uniques]

    for i in trange(n_t, desc="Training timesteps"):
        # Fresh noise each timestep
        X0 = np.random.normal(size=X1.shape)

        t = torch.ones(X0.shape[0]) * t_levels[i]
        _, xt, ut = FM.sample_location_and_conditional_flow(
            torch.from_numpy(X0), torch.from_numpy(X1), t=t
        )
        xt_np, ut_np = xt.numpy(), ut.numpy()

        # Train regressors for this timestep
        for ji, j in enumerate(y_uniques):
            mask = mask_y[j]
            X_masked = xt_np[mask, :]
            y_masked = ut_np[mask, :]

            if multi_output:
                model = xgb.XGBRegressor(
                    n_estimators=args.n_estimators,
                    objective="reg:squarederror",
                    eta=args.eta,
                    max_depth=args.max_depth,
                    n_jobs=args.n_threads,
                    reg_lambda=args.reg_lambda,
                    reg_alpha=args.reg_alpha,
                    subsample=args.subsample,
                    seed=666,
                    tree_method=args.tree_method,
                    device="cpu",
                    multi_strategy="multi_output_tree",
                )
                model.fit(X_masked, y_masked)
                regr_mo[ji][i] = model
            else:
                for k in range(c):
                    model = xgb.XGBRegressor(
                        n_estimators=args.n_estimators,
                        objective="reg:squarederror",
                        eta=args.eta,
                        max_depth=args.max_depth,
                        n_jobs=args.n_threads,
                        reg_lambda=args.reg_lambda,
                        reg_alpha=args.reg_alpha,
                        subsample=args.subsample,
                        seed=666,
                        tree_method=args.tree_method,
                        device="cpu",
                    )
                    model.fit(X_masked, y_masked[:, k])
                    regr[ji][i][k] = model

        # xt_np, ut_np are discarded here — memory freed

    if multi_output:
        return regr_mo
    return regr


# Legacy non-batched functions for backward compatibility
def prepare_training_data(X, y, duplicate_K, n_t, flow_type, sigma):
    """Duplicate, scale, and build (xt, ut) pairs at each timestep."""
    b, c = X.shape

    # Class masks
    y_uniques, y_counts = np.unique(y, return_counts=True)
    y_probs = y_counts / y_counts.sum()
    mask_y = {}
    for label in y_uniques:
        m = (y == label)
        mask_y[label] = np.tile(m, duplicate_K)

    # MinMax scaling to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Duplicate
    X1 = np.tile(X_scaled, (duplicate_K, 1))
    X0 = np.random.normal(size=X1.shape)

    # Flow matching
    FM = build_flow_matcher(flow_type, sigma)
    t_levels = np.linspace(1e-3, 1, num=n_t)

    X_train = np.zeros((n_t, X0.shape[0], X0.shape[1]))
    y_train = np.zeros((n_t, X0.shape[0], X0.shape[1]))

    for i in trange(n_t, desc="Building flow pairs"):
        t = torch.ones(X0.shape[0]) * t_levels[i]
        _, xt, ut = FM.sample_location_and_conditional_flow(
            torch.from_numpy(X0), torch.from_numpy(X1), t=t
        )
        X_train[i], y_train[i] = xt.numpy(), ut.numpy()

    return X_train, y_train, scaler, mask_y, y_uniques, y_probs


def train_models(X_train, y_train, mask_y, y_uniques, n_t, c, args):
    """Train per-timestep, per-class, per-feature XGBoost regressors."""
    b_dup = X_train.shape[1]

    def _train_one(X, y_col):
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            objective="reg:squarederror",
            eta=args.eta,
            max_depth=args.max_depth,
            n_jobs=args.n_threads,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            subsample=args.subsample,
            seed=666,
            tree_method=args.tree_method,
            device="cpu",
        )
        model.fit(X, y_col)
        return model

    regr = Parallel(n_jobs=args.n_jobs)(
        delayed(_train_one)(
            X_train[i][mask_y[j], :],
            y_train[i][mask_y[j], k],
        )
        for i in trange(n_t, desc="Training timesteps")
        for k in range(c)
        for j in y_uniques
    )

    # Reorganise into regr[class_idx][timestep][feature]
    n_y = len(y_uniques)
    regr_ = [[[None for _ in range(c)] for _ in range(n_t)] for _ in y_uniques]
    idx = 0
    for i in range(n_t):
        for k in range(c):
            for ji, j in enumerate(y_uniques):
                regr_[ji][i][k] = regr[idx]
                idx += 1
    return regr_


# ── Sampling ──────────────────────────────────────────────────────────

def build_model_fn(regr, y_uniques, c, n_t, mask_y, multi_output=False):
    """Return a callable  my_model(t, xt) → velocity  for ODE solvers.

    Supports both per-feature models (regr[class][timestep][feature])
    and multi-output models (regr[class][timestep] → predicts all features).
    """

    def my_model(t, xt, mask_y=None):
        xt = xt.reshape(xt.shape[0] // c, c)
        out = np.zeros(xt.shape)
        i = int(round(t * (n_t - 1)))
        for j, label in enumerate(y_uniques):
            m = mask_y[label]
            if multi_output:
                out[m, :] = regr[j][i].predict(xt[m, :])
            else:
                for k in range(c):
                    out[m, k] = regr[j][i][k].predict(xt[m, :])
        return out.reshape(-1)

    return partial(my_model, mask_y=mask_y)


def sample(regr, y_uniques, y_probs, c, n_t, scaler, X_min, X_max, solver, solver_steps, n_samples, multi_output=False):
    """Generate samples from trained flowBDT models."""
    x0 = np.random.normal(size=(n_samples, c))

    # Random class labels
    label_y = y_uniques[np.argmax(
        np.random.multinomial(1, y_probs, size=n_samples), axis=1
    )]
    mask_y_fake = {}
    for label in y_uniques:
        mask_y_fake[label] = (label_y == label)

    model_fn = build_model_fn(regr, y_uniques, c, n_t, mask_y_fake, multi_output=multi_output)

    solvers = {
        "euler": euler_solve,
        "midpoint": midpoint_solve,
        "dopri5": dpori5_solve_numpy,
    }
    solve = solvers[solver]

    print(f"Sampling {n_samples} events with {solver} ({solver_steps} steps)...")
    t0 = time.time()
    solution = solve(x0=x0.reshape(-1), my_model=model_fn, N=solver_steps)
    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.3f}s ({elapsed / n_samples * 1000:.3f} ms/event)")

    solution = solution.reshape(n_samples, c)

    # Inverse MinMax, clip to data range
    solution = scaler.inverse_transform(solution)
    solution = np.clip(solution, X_min, X_max)

    return solution, label_y


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse features
    features = args.features.split(",") if args.features else None

    # Load
    print("Loading data...")
    X_raw, y = load_data(args.data, features)
    print(f"  Loaded {X_raw.shape[0]} events, {X_raw.shape[1]} features")

    # Optionally strip derived features for training
    derived_info = None
    if args.strip_derived and X_raw.shape[1] == 12:
        X, derived_info = strip_derived_features(X_raw)
        print(f"  Stripped derived features: {X_raw.shape[1]} → {X.shape[1]} independent features")
    else:
        X = X_raw

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    b, c = X.shape

    # Shuffle
    perm = np.random.permutation(b)
    X, y = X[perm], y[perm]

    # Scale and prepare
    print("Preparing scaling...")
    X_scaled, scaler, mask_y, y_uniques, y_probs = prepare_scaling(
        X, y, args.duplicate_k
    )

    # Train (batched — one timestep at a time, 30x less memory)
    mode = "multi-output" if args.multi_output else "per-feature"
    print(f"Training XGBoost regressors ({mode}, {args.flow_type})...")
    t0 = time.time()
    regr = train_batched(
        X_scaled, y, args.duplicate_k, args.n_timesteps,
        args.flow_type, args.sigma, mask_y, y_uniques, c, args,
    )
    print(f"Training done in {time.time() - t0:.1f}s")

    # Save models + scaler
    with open(os.path.join(args.output_dir, "models.pkl"), "wb") as f:
        pickle.dump(regr, f)
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "X_min": X_min, "X_max": X_max,
                      "y_uniques": y_uniques, "y_probs": y_probs,
                      "c": c, "n_t": args.n_timesteps,
                      "multi_output": args.multi_output,
                      "derived_info": derived_info}, f)

    # Sample
    n_samples = args.n_samples if args.n_samples > 0 else b
    solution, labels = sample(
        regr, y_uniques, y_probs, c, args.n_timesteps,
        scaler, X_min, X_max, args.solver, args.solver_steps, n_samples,
        multi_output=args.multi_output,
    )

    # Restore derived features if stripped
    if derived_info is not None:
        print("Restoring derived features (tau21, tau32, d2_obs) from independent features...")
        solution = restore_derived_features(solution, derived_info)
        print(f"  Output shape: {solution.shape}")

    # Cholesky correlation correction
    if args.cholesky:
        print("Applying Cholesky correlation correction...")
        if args.real_data_for_cholesky:
            real_ref = np.load(args.real_data_for_cholesky)
        else:
            real_ref = X_raw
        solution = cholesky_correction(solution, real_ref)

    np.save(os.path.join(args.output_dir, "generated_samples.npy"), solution)
    np.save(os.path.join(args.output_dir, "generated_labels.npy"), labels)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
