"""Time-conditioned flowBDT training for improved correlation modeling.

Instead of training n_t separate model sets (one per timestep), trains a
SINGLE multi-output XGBoost model per class that takes [x_t, t] as input.
This is analogous to how neural networks condition on time via embeddings.

Optionally adds pairwise feature interactions (x_i * x_j) as extra input
features to help trees learn cross-feature correlations.

Usage examples:
  # Basic time-conditioned (single model, all timesteps pooled)
  uv run python scripts/train_time_conditioned.py --data data/jetnet/t_hlv12.npy

  # With feature interactions for better correlations
  uv run python scripts/train_time_conditioned.py --data data/jetnet/t_hlv12.npy --feature-interactions

  # Higher capacity + Cholesky post-correction
  uv run python scripts/train_time_conditioned.py --data data/jetnet/t_hlv12.npy \\
      --max-depth 8 --n-estimators 500 --eta 0.02 --cholesky --cholesky-alpha 0.3

  # Compare with per-timestep baseline
  uv run python scripts/train_time_conditioned.py --data data/jetnet/t_hlv12.npy \\
      --per-timestep-baseline --output-dir results/per_timestep_baseline/
"""
import argparse
import importlib
import json
import os
import pickle
import sys
import time
import types

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Make repo root importable as both top-level modules and as "BUFF.*"
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
if "BUFF" not in sys.modules:
    sys.modules["BUFF"] = importlib.import_module(os.path.basename(_repo_root).replace("-", "_") if False else "__init__")
    # Register repo root as BUFF package
    _buff = types.ModuleType("BUFF")
    _buff.__path__ = [_repo_root]
    _buff.__file__ = os.path.join(_repo_root, "__init__.py")
    sys.modules["BUFF"] = _buff

from runner.train_and_sample import (
    build_model_fn, build_model_fn_time_conditioned,
    cholesky_correction, load_data, prepare_scaling,
    restore_derived_features, strip_derived_features,
    train_batched, train_time_conditioned,
    JETNET_INDEPENDENT_INDICES,
)
from runner.ode_example import euler_solve, midpoint_solve, dpori5_solve_numpy
from evaluation.discriminator import train_discriminator
from evaluation.consistency import tau21_consistency_check, tau32_consistency_check


def parse_args():
    p = argparse.ArgumentParser(description="Time-conditioned flowBDT training")
    p.add_argument("--data", default="data/jetnet/t_hlv12.npy")
    p.add_argument("--n-timesteps", type=int, default=30)
    p.add_argument("--duplicate-k", type=int, default=20)
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--solver", choices=["euler", "midpoint", "dopri5"], default="euler")
    p.add_argument("--n-threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1980)

    # XGBoost hyperparameters (higher capacity defaults for time-conditioned)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.5)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9, dest="colsample_bytree")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda"],
                   help="XGBoost device: 'cuda' for GPU training (much faster)")
    p.add_argument("--max-total-samples", type=int, default=3_000_000,
                   help="Max total training samples across all timesteps (memory budget)")

    # Strategy flags
    p.add_argument("--feature-interactions", action="store_true",
                   help="Add pairwise feature products as extra inputs")
    p.add_argument("--cholesky", action="store_true",
                   help="Apply Cholesky correlation correction post-generation")
    p.add_argument("--cholesky-alpha", type=float, default=0.3)
    p.add_argument("--per-timestep-baseline", action="store_true",
                   help="Train per-timestep multi-output models for comparison")

    p.add_argument("--output-dir", default="results/time_conditioned/")
    return p.parse_args()


class ArgsAdapter:
    """Adapter to pass CLI args to train_* functions."""
    pass


def evaluate(real_12, gen_12):
    """Evaluate generated samples against real data."""
    from scipy.stats import wasserstein_distance

    # AUC on 9 independent features
    r9 = train_discriminator(
        real_12[:, JETNET_INDEPENDENT_INDICES],
        gen_12[:, JETNET_INDEPENDENT_INDICES],
    )
    # AUC on all 12
    r12 = train_discriminator(real_12, gen_12)

    # Correlation error
    corr_real = np.corrcoef(real_12.T)
    corr_gen = np.corrcoef(gen_12.T)
    diff = np.abs(corr_real - corr_gen)
    np.fill_diagonal(diff, 0)
    mask_tri = np.triu(np.ones_like(diff, dtype=bool), k=1)

    # Consistency
    c_tau21 = tau21_consistency_check(gen_12[:, 4], gen_12[:, 5], gen_12[:, 7])
    c_tau32 = tau32_consistency_check(gen_12[:, 5], gen_12[:, 6], gen_12[:, 8])

    # Per-feature W1
    features = ["d12", "d2", "mass", "pt", "tau1", "tau2", "tau3",
                "tau21", "tau32", "ecf2", "ecf3", "d2_obs"]
    w1s = {}
    for i, f in enumerate(features):
        w1s[f] = float(wasserstein_distance(real_12[:, i], gen_12[:, i]))

    metrics = {
        "auc_9": r9["auc_test"],
        "auc_12": r12["auc_test"],
        "mean_delta_rho": float(diff[mask_tri].mean()),
        "max_delta_rho": float(diff[mask_tri].max()),
        "consistency_tau21_w1": c_tau21["w1"],
        "consistency_tau32_w1": c_tau32["w1"],
        "w1_per_feature": w1s,
    }

    print(f"\n{'='*60}")
    print(f"AUC (9 indep):   {metrics['auc_9']:.4f}")
    print(f"AUC (12 feat):   {metrics['auc_12']:.4f}")
    print(f"Mean |Δρ|:       {metrics['mean_delta_rho']:.4f}")
    print(f"Max  |Δρ|:       {metrics['max_delta_rho']:.4f}")
    print(f"Consistency tau21 W1: {metrics['consistency_tau21_w1']:.4f}")
    print(f"Consistency tau32 W1: {metrics['consistency_tau32_w1']:.4f}")
    print(f"\nPer-feature W1:")
    for f, w in w1s.items():
        print(f"  {f:>8}: {w:.6f}")
    print(f"{'='*60}\n")

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load and strip derived features
    X_raw = np.load(args.data)
    print(f"Loaded {X_raw.shape}")
    X, derived_info = strip_derived_features(X_raw)
    print(f"Stripped to {X.shape[1]} independent features")

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    b, c = X.shape

    perm = np.random.permutation(b)
    X, y = X[perm], np.zeros(b)

    X_scaled, scaler, mask_y, y_uniques, y_probs = prepare_scaling(
        X, y, args.duplicate_k
    )

    # Build args adapter for XGBoost
    train_a = ArgsAdapter()
    train_a.n_estimators = args.n_estimators
    train_a.eta = args.eta
    train_a.max_depth = args.max_depth
    train_a.n_threads = args.n_threads
    train_a.reg_lambda = args.reg_lambda
    train_a.reg_alpha = args.reg_alpha
    train_a.subsample = args.subsample
    train_a.colsample_bytree = args.colsample_bytree
    train_a.tree_method = "hist"
    train_a.device = args.device
    train_a.multi_output = True  # always multi-output for time-conditioned
    train_a.max_total_samples = args.max_total_samples

    n_t = args.n_timesteps

    # ── Train ──
    if args.per_timestep_baseline:
        print(f"\n=== Per-timestep multi-output baseline ===")
        t0 = time.time()
        regr = train_batched(
            X_scaled, y, args.duplicate_k, n_t, "icfm", 0.0,
            mask_y, y_uniques, c, train_a,
        )
        train_time_s = time.time() - t0
        print(f"Training: {train_time_s:.0f}s")

        # Sample
        n_samples = b
        x0 = np.random.normal(size=(n_samples, c))
        label_y = y_uniques[np.argmax(
            np.random.multinomial(1, y_probs, size=n_samples), axis=1
        )]
        mask_y_fake = {label: (label_y == label) for label in y_uniques}

        model_fn = build_model_fn(regr, y_uniques, c, n_t, mask_y_fake, multi_output=True)
        method_name = "per_timestep_mo"
    else:
        print(f"\n=== Time-conditioned model ===")
        print(f"  Hyperparameters: depth={args.max_depth}, n_est={args.n_estimators}, "
              f"eta={args.eta}, interactions={args.feature_interactions}")
        t0 = time.time()
        models, input_dim = train_time_conditioned(
            X_scaled, y, args.duplicate_k, n_t, "icfm", 0.0,
            mask_y, y_uniques, c, train_a,
            feature_interactions=args.feature_interactions,
        )
        train_time_s = time.time() - t0
        print(f"Training: {train_time_s:.0f}s")

        # Sample
        n_samples = b
        x0 = np.random.normal(size=(n_samples, c))
        label_y = y_uniques[np.argmax(
            np.random.multinomial(1, y_probs, size=n_samples), axis=1
        )]
        mask_y_fake = {label: (label_y == label) for label in y_uniques}

        model_fn = build_model_fn_time_conditioned(
            models, y_uniques, c, n_t, mask_y_fake,
            feature_interactions=args.feature_interactions,
        )
        method_name = "time_cond" + ("_interact" if args.feature_interactions else "")

    solvers = {"euler": euler_solve, "midpoint": midpoint_solve, "dopri5": dpori5_solve_numpy}
    solve = solvers[args.solver]

    print(f"Sampling {n_samples} events with {args.solver} ({args.solver_steps} steps)...")
    t0 = time.time()
    solution = solve(x0=x0.reshape(-1), my_model=model_fn, N=args.solver_steps)
    sample_time_s = time.time() - t0
    print(f"Sampling: {sample_time_s:.1f}s")

    solution = solution.reshape(n_samples, c)
    solution = scaler.inverse_transform(solution)
    solution = np.clip(solution, X_min, X_max)

    # Cholesky correction
    if args.cholesky:
        print(f"Applying Cholesky correction (alpha={args.cholesky_alpha})...")
        # Re-load unstripped data for reference
        X_ref, _ = strip_derived_features(np.load(args.data))
        solution = cholesky_correction(solution, X_ref, alpha=args.cholesky_alpha)

    # Restore derived features
    solution_full = restore_derived_features(solution, derived_info)
    print(f"Restored to {solution_full.shape[1]} features")

    np.save(os.path.join(args.output_dir, "generated_samples.npy"), solution_full)

    # Evaluate
    X_raw_orig = np.load(args.data)
    metrics = evaluate(X_raw_orig, solution_full)
    metrics["method"] = method_name
    metrics["train_time_s"] = train_time_s
    metrics["sample_time_s"] = sample_time_s
    metrics["hyperparameters"] = {
        "max_depth": args.max_depth, "n_estimators": args.n_estimators,
        "eta": args.eta, "n_timesteps": n_t, "duplicate_k": args.duplicate_k,
        "solver": args.solver, "solver_steps": args.solver_steps,
        "feature_interactions": args.feature_interactions,
        "cholesky": args.cholesky, "cholesky_alpha": args.cholesky_alpha,
    }

    with open(os.path.join(args.output_dir, "quick_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
