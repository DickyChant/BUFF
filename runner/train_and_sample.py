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

try:
    from BUFF.runner.ode_example import (
        dpori5_solve_numpy, euler_solve, midpoint_solve,
    )
except ImportError:
    from runner.ode_example import (
        dpori5_solve_numpy, euler_solve, midpoint_solve,
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

def cholesky_correction(gen, real, alpha=1.0):
    """Match the correlation structure of generated data to real data.

    Preserves marginal means/stds while correcting the correlation matrix.
    When alpha < 1.0, blends between original and fully corrected:
        result = (1 - alpha) * gen + alpha * corrected
    This is useful to avoid extreme tail distortions that can amplify
    through derived features (e.g. tau32 = tau3/tau2).
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

    if alpha < 1.0:
        return (1 - alpha) * gen + alpha * gen_corrected
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
    p.add_argument("--colsample-bytree", type=float, default=1.0, dest="colsample_bytree")
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
        "--cholesky-alpha", type=float, default=1.0,
        help="Blending factor for Cholesky correction (0=none, 1=full, 0.3 recommended)",
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

            colsample = getattr(args, 'colsample_bytree', 1.0)
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
                    colsample_bytree=colsample,
                    seed=666,
                    tree_method=args.tree_method,
                    device=getattr(args, 'device', 'cpu'),
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
                        colsample_bytree=colsample,
                        seed=666,
                        tree_method=args.tree_method,
                        device=getattr(args, 'device', 'cpu'),
                    )
                    model.fit(X_masked, y_masked[:, k])
                    regr[ji][i][k] = model

        # xt_np, ut_np are discarded here — memory freed

    if multi_output:
        return regr_mo
    return regr


def train_batched_with_residuals(X_scaled, y, duplicate_K, n_t, flow_type, sigma,
                                  mask_y, y_uniques, c, args, pass2_args=None):
    """Two-pass training: pass-1 per-feature models + pass-2 residual correction.

    Pass 2 trains on augmented input [xt, v_hat] to predict (true_v - v_hat),
    giving each feature's model access to what other features predicted.

    Returns (pass1_models, pass2_models).
    """
    FM = build_flow_matcher(flow_type, sigma)
    t_levels = np.linspace(1e-3, 1, num=n_t)
    X1 = np.tile(X_scaled, (duplicate_K, 1))

    if pass2_args is None:
        pass2_args = type(args)()
        for k, v in vars(args).items():
            setattr(pass2_args, k, v)
        pass2_args.max_depth = 4
        pass2_args.n_estimators = 100
        pass2_args.eta = 0.1

    # Pass 1: standard per-feature models
    regr_p1 = [[[None for _ in range(c)] for _ in range(n_t)] for _ in y_uniques]
    # Pass 2: residual models with augmented input
    regr_p2 = [[[None for _ in range(c)] for _ in range(n_t)] for _ in y_uniques]

    colsample = getattr(args, 'colsample_bytree', 1.0)
    colsample_p2 = getattr(pass2_args, 'colsample_bytree', 1.0)

    for i in trange(n_t, desc="Training (two-pass)"):
        X0 = np.random.normal(size=X1.shape)
        t = torch.ones(X0.shape[0]) * t_levels[i]
        _, xt, ut = FM.sample_location_and_conditional_flow(
            torch.from_numpy(X0), torch.from_numpy(X1), t=t
        )
        xt_np, ut_np = xt.numpy(), ut.numpy()

        # --- Pass 1: train per-feature models ---
        for ji, j in enumerate(y_uniques):
            mask = mask_y[j]
            X_masked = xt_np[mask, :]
            y_masked = ut_np[mask, :]
            for k in range(c):
                model = xgb.XGBRegressor(
                    n_estimators=args.n_estimators, objective="reg:squarederror",
                    eta=args.eta, max_depth=args.max_depth, n_jobs=args.n_threads,
                    reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
                    subsample=args.subsample, colsample_bytree=colsample,
                    seed=666, tree_method=args.tree_method, device=getattr(args, 'device', 'cpu'),
                )
                model.fit(X_masked, y_masked[:, k])
                regr_p1[ji][i][k] = model

        # --- Pass 2: predict pass-1 velocities, train residual models ---
        for ji, j in enumerate(y_uniques):
            mask = mask_y[j]
            X_masked = xt_np[mask, :]
            y_masked = ut_np[mask, :]

            # Get pass-1 predictions
            v_hat = np.zeros_like(y_masked)
            for k in range(c):
                v_hat[:, k] = regr_p1[ji][i][k].predict(X_masked)

            # Augmented input: [xt, v_hat]
            X_aug = np.hstack([X_masked, v_hat])
            residual = y_masked - v_hat

            for k in range(c):
                model = xgb.XGBRegressor(
                    n_estimators=pass2_args.n_estimators, objective="reg:squarederror",
                    eta=pass2_args.eta, max_depth=pass2_args.max_depth,
                    n_jobs=pass2_args.n_threads,
                    reg_lambda=pass2_args.reg_lambda, reg_alpha=pass2_args.reg_alpha,
                    subsample=pass2_args.subsample, colsample_bytree=colsample_p2,
                    seed=666, tree_method=pass2_args.tree_method, device=getattr(pass2_args, 'device', 'cpu'),
                )
                model.fit(X_aug, residual[:, k])
                regr_p2[ji][i][k] = model

    return regr_p1, regr_p2


# ── Time-conditioned training ─────────────────────────────────────

def _make_feature_interactions(X):
    """Append pairwise product features to X. Returns (X_aug, n_interact)."""
    n_feat = X.shape[1]
    interactions = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            interactions.append(X[:, i] * X[:, j])
    if not interactions:
        return X, 0
    interact_mat = np.column_stack(interactions)
    return np.hstack([X, interact_mat]), len(interactions)


def train_time_conditioned(X_scaled, y, duplicate_K, n_t, flow_type, sigma,
                           mask_y, y_uniques, c, args, feature_interactions=False):
    """Train a single multi-output XGBoost per class across ALL timesteps.

    Instead of n_t separate models, pools all timestep data together with t
    appended as an extra input feature: input = [x_t, t] → output = v_t.
    This is analogous to how neural networks condition on time.

    Memory-efficient: subsamples each timestep's data to keep total dataset
    size manageable (controlled by max_samples_per_timestep).

    Returns: (models_dict, input_dim) where models_dict[class_idx] = one XGBoost model,
    and input_dim is the augmented input dimension (for inference).
    """
    FM = build_flow_matcher(flow_type, sigma)
    t_levels = np.linspace(1e-3, 1, num=n_t)
    X1 = np.tile(X_scaled, (duplicate_K, 1))

    # Memory budget: limit total training samples to ~3M rows to stay under ~2GB
    # With N=178k, K=20, that's 3.56M per timestep × 30 = 107M total — way too much.
    # Subsample each timestep to keep total manageable.
    max_total_samples = getattr(args, 'max_total_samples', 3_000_000)
    max_per_timestep = max_total_samples // n_t
    n_dup = X1.shape[0]

    # Collect flow pairs from all timesteps (subsampled)
    all_xt = {ji: [] for ji in range(len(y_uniques))}
    all_ut = {ji: [] for ji in range(len(y_uniques))}

    for i in trange(n_t, desc="Building flow pairs (all timesteps)"):
        X0 = np.random.normal(size=X1.shape)
        t = torch.ones(X0.shape[0]) * t_levels[i]
        _, xt, ut = FM.sample_location_and_conditional_flow(
            torch.from_numpy(X0), torch.from_numpy(X1), t=t
        )
        xt_np, ut_np = xt.numpy(), ut.numpy()

        # Subsample if needed to stay within memory budget
        if xt_np.shape[0] > max_per_timestep:
            idx = np.random.choice(xt_np.shape[0], max_per_timestep, replace=False)
            xt_np = xt_np[idx]
            ut_np = ut_np[idx]

        # Append time as extra column
        t_col = np.full((xt_np.shape[0], 1), t_levels[i])
        xt_with_t = np.hstack([xt_np, t_col])

        for ji, j in enumerate(y_uniques):
            mask = mask_y[j]
            # Mask must be applied to subsampled indices
            if xt_with_t.shape[0] < len(mask):
                # Subsampled — mask doesn't align, just use all rows
                # (subsampling already mixed classes proportionally)
                all_xt[ji].append(xt_with_t)
                all_ut[ji].append(ut_np)
            else:
                all_xt[ji].append(xt_with_t[mask, :])
                all_ut[ji].append(ut_np[mask, :])

        # Free intermediate arrays
        del X0, xt, ut, xt_np, ut_np

    # Concatenate all timesteps and train one model per class
    models = {}
    colsample = getattr(args, 'colsample_bytree', 1.0)

    for ji, j in enumerate(y_uniques):
        X_all = np.vstack(all_xt[ji])
        y_all = np.vstack(all_ut[ji])

        # Free the lists immediately after concatenation
        del all_xt[ji], all_ut[ji]

        if feature_interactions:
            X_all, n_interact = _make_feature_interactions(X_all)
            print(f"  Class {j}: added {n_interact} interaction features → {X_all.shape[1]} input dims")

        print(f"  Class {j}: training on {X_all.shape[0]} samples, "
              f"{X_all.shape[1]} input features → {y_all.shape[1]} outputs")

        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            objective="reg:squarederror",
            eta=args.eta,
            max_depth=args.max_depth,
            n_jobs=args.n_threads,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            subsample=args.subsample,
            colsample_bytree=colsample,
            seed=666,
            tree_method=args.tree_method,
            device=getattr(args, 'device', 'cpu'),
            multi_strategy="multi_output_tree",
        )
        model.fit(X_all, y_all)
        models[ji] = model

        # Free training data after fitting
        del X_all, y_all

    input_dim = c + 1  # features + time
    if feature_interactions:
        # Recompute from dimensions
        input_dim = (c + 1) + (c + 1) * c // 2

    return models, input_dim


def train_reflow_student(X0, X1_teacher, y, y_uniques, args,
                         predict_endpoint=False, feature_interactions=False):
    """Train a 1-step student on teacher-generated endpoint pairs.

    The student operates directly in the scaled BUFF model space.
    When predict_endpoint=False, it learns the displacement X1 - X0 so a
    single Euler update with h=1 lands on the teacher endpoint.
    """
    c = X0.shape[1]
    target = X1_teacher if predict_endpoint else (X1_teacher - X0)

    models = {}
    colsample = getattr(args, 'colsample_bytree', 1.0)

    for ji, label in enumerate(y_uniques):
        mask = (y == label)
        X_masked = X0[mask, :]
        y_masked = target[mask, :]

        if feature_interactions:
            X_masked, n_interact = _make_feature_interactions(X_masked)
            print(f"  Class {label}: added {n_interact} interaction features "
                  f"→ {X_masked.shape[1]} input dims")

        print(f"  Class {label}: distilling {X_masked.shape[0]} pairs, "
              f"{X_masked.shape[1]} input features → {y_masked.shape[1]} outputs")

        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            objective="reg:squarederror",
            eta=args.eta,
            max_depth=args.max_depth,
            n_jobs=args.n_threads,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            subsample=args.subsample,
            colsample_bytree=colsample,
            seed=666,
            tree_method=args.tree_method,
            device=getattr(args, 'device', 'cpu'),
            multi_strategy="multi_output_tree",
        )
        model.fit(X_masked, y_masked)
        models[ji] = model

    input_dim = c
    if feature_interactions:
        input_dim = c + c * (c - 1) // 2

    return models, input_dim


def build_model_fn_time_conditioned(models, y_uniques, c, n_t, mask_y,
                                     feature_interactions=False):
    """Model function for time-conditioned single-model inference.

    At each ODE step, appends the current time t as an extra input column
    and queries the single model for that class.
    """
    def my_model(t, xt, mask_y=None):
        xt = xt.reshape(xt.shape[0] // c, c)
        out = np.zeros(xt.shape)
        n = xt.shape[0]

        # Append time column
        t_col = np.full((n, 1), t)
        xt_with_t = np.hstack([xt, t_col])

        if feature_interactions:
            xt_with_t, _ = _make_feature_interactions(xt_with_t)

        for j, label in enumerate(y_uniques):
            m = mask_y[label]
            out[m, :] = models[j].predict(xt_with_t[m, :])
        return out.reshape(-1)

    return partial(my_model, mask_y=mask_y)


def build_model_fn_one_step(models, y_uniques, c, mask_y,
                            predict_endpoint=False, feature_interactions=False):
    """Model function for a distilled 1-step student.

    The returned callable matches the BUFF ODE-solver interface, so a
    one-step Euler solve (`N=2`) can reuse the existing sampling code.
    """
    def my_model(t, xt, mask_y=None):
        del t  # The distilled student is time-independent.
        xt = xt.reshape(xt.shape[0] // c, c)
        out = np.zeros(xt.shape)

        X_eval = xt
        if feature_interactions:
            X_eval, _ = _make_feature_interactions(X_eval)

        for j, label in enumerate(y_uniques):
            m = mask_y[label]
            if not np.any(m):
                continue
            pred = models[j].predict(X_eval[m, :])
            if predict_endpoint:
                out[m, :] = pred - xt[m, :]
            else:
                out[m, :] = pred
        return out.reshape(-1)

    return partial(my_model, mask_y=mask_y)


def build_model_fn_residual(regr_p1, regr_p2, y_uniques, c, n_t, mask_y):
    """Model function for two-pass residual stacking.

    At each evaluation: predict pass-1 velocity, stack [xt, v_hat],
    predict pass-2 residual, return v_hat + residual.
    """
    def my_model(t, xt, mask_y=None):
        xt = xt.reshape(xt.shape[0] // c, c)
        out = np.zeros(xt.shape)
        i = int(round(t * (n_t - 1)))
        for j, label in enumerate(y_uniques):
            m = mask_y[label]
            X_m = xt[m, :]
            # Pass 1
            v_hat = np.zeros((X_m.shape[0], c))
            for k in range(c):
                v_hat[:, k] = regr_p1[j][i][k].predict(X_m)
            # Pass 2: augmented input
            X_aug = np.hstack([X_m, v_hat])
            for k in range(c):
                v_hat[:, k] += regr_p2[j][i][k].predict(X_aug)
            out[m, :] = v_hat
        return out.reshape(-1)

    return partial(my_model, mask_y=mask_y)


def build_model_fn_ensemble(model_fns):
    """Average predictions from multiple model functions."""
    n = len(model_fns)

    def my_model(t, xt):
        total = model_fns[0](t=t, xt=xt)
        for fn in model_fns[1:]:
            total = total + fn(t=t, xt=xt)
        return total / n

    return my_model


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
            device=getattr(args, 'device', 'cpu'),
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


def sample_one_step(models, y_uniques, y_probs, c, scaler, X_min, X_max,
                    n_samples, predict_endpoint=False, feature_interactions=False):
    """Generate samples from a distilled 1-step student."""
    x0 = np.random.normal(size=(n_samples, c))

    label_y = y_uniques[np.argmax(
        np.random.multinomial(1, y_probs, size=n_samples), axis=1
    )]
    mask_y_fake = {}
    for label in y_uniques:
        mask_y_fake[label] = (label_y == label)

    model_fn = build_model_fn_one_step(
        models, y_uniques, c, mask_y_fake,
        predict_endpoint=predict_endpoint,
        feature_interactions=feature_interactions,
    )

    print(f"Sampling {n_samples} events with one-step Euler...")
    t0 = time.time()
    solution = euler_solve(x0=x0.reshape(-1), my_model=model_fn, N=2)
    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.3f}s ({elapsed / n_samples * 1000:.3f} ms/event)")

    solution = solution.reshape(n_samples, c)
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

    # Cholesky correlation correction (applied BEFORE restoring derived features
    # so that functional relationships like tau21=tau2/tau1 are preserved)
    if args.cholesky:
        alpha = args.cholesky_alpha
        print(f"Applying Cholesky correlation correction (alpha={alpha})...")
        if args.real_data_for_cholesky:
            real_ref = np.load(args.real_data_for_cholesky)
        else:
            real_ref = X  # Use the (possibly stripped) training data as reference
        solution = cholesky_correction(solution, real_ref, alpha=alpha)

    # Restore derived features if stripped
    if derived_info is not None:
        print("Restoring derived features (tau21, tau32, d2_obs) from independent features...")
        solution = restore_derived_features(solution, derived_info)
        print(f"  Output shape: {solution.shape}")

    np.save(os.path.join(args.output_dir, "generated_samples.npy"), solution)
    np.save(os.path.join(args.output_dir, "generated_labels.npy"), labels)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
