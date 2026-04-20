"""Unified improved training script for BUFF flowBDT.

Combines four improvement strategies:
  A) Higher capacity models with proper regularization
  B) Cholesky correction on 9 independent features (preserves derived relationships)
  C) Residual stacking (two-pass training for cross-feature coupling)
  D) Model ensemble (multiple seeds, averaged velocities)

Usage examples:
  # A1 + Cholesky (fast baseline, ~250 min)
  python scripts/train_improved.py --config a1 --cholesky-9feat

  # A1 + residual stacking + Cholesky (~400 min)
  python scripts/train_improved.py --config a1 --residual-stacking --cholesky-9feat

  # A1 + ensemble(3 seeds) + Cholesky (~750 min)
  python scripts/train_improved.py --config a1 --ensemble-seeds 1980,2024,42 --cholesky-9feat

  # All combined
  python scripts/train_improved.py --config a1 --residual-stacking --ensemble-seeds 1980,2024,42 --cholesky-9feat
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
try:
    from BUFF.runner.train_and_sample import (
        build_model_fn, build_model_fn_ensemble, build_model_fn_residual,
        cholesky_correction, load_data, prepare_scaling,
        restore_derived_features, strip_derived_features,
        train_batched, train_batched_with_residuals, JETNET_INDEPENDENT_INDICES,
    )
    from BUFF.runner.ode_example import euler_solve
    from BUFF.evaluation.discriminator import train_discriminator
    from BUFF.evaluation.consistency import tau21_consistency_check, tau32_consistency_check
except ImportError:
    from runner.train_and_sample import (
        build_model_fn, build_model_fn_ensemble, build_model_fn_residual,
        cholesky_correction, load_data, prepare_scaling,
        restore_derived_features, strip_derived_features,
        train_batched, train_batched_with_residuals, JETNET_INDEPENDENT_INDICES,
    )
    from runner.ode_example import euler_solve
    from evaluation.discriminator import train_discriminator
    from evaluation.consistency import tau21_consistency_check, tau32_consistency_check

# ── Hyperparameter presets ──────────────────────────────────────────
CONFIGS = {
    "a1": dict(
        max_depth=6, n_estimators=300, eta=0.03,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.5,
    ),
    "a2": dict(
        max_depth=7, n_estimators=400, eta=0.02,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, reg_alpha=0.5,
    ),
    "baseline": dict(
        max_depth=4, n_estimators=100, eta=0.1,
        subsample=1.0, colsample_bytree=1.0,
        reg_lambda=0.1, reg_alpha=0.2,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Improved BUFF training pipeline")
    p.add_argument("--data", default="data/jetnet/t_hlv12.npy")
    p.add_argument("--config", choices=list(CONFIGS.keys()), default="a1")
    p.add_argument("--n-timesteps", type=int, default=30)
    p.add_argument("--duplicate-k", type=int, default=20)
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--n-threads", type=int, default=16)
    p.add_argument("--residual-stacking", action="store_true",
                   help="Enable two-pass residual stacking (Strategy C)")
    p.add_argument("--ensemble-seeds", type=str, default=None,
                   help="Comma-separated seeds for ensemble (Strategy D)")
    p.add_argument("--cholesky-9feat", action="store_true",
                   help="Cholesky correction on 9 independent features (Strategy B)")
    p.add_argument("--cholesky-alpha", type=float, default=0.3,
                   help="Blending factor for Cholesky (0=none, 1=full, 0.3=recommended)")
    p.add_argument("--output-dir", default="results/improved/")
    return p.parse_args()


def make_args_obj(config_dict, n_threads=16):
    """Create a mock args object from config dict."""
    class Args:
        pass
    a = Args()
    for k, v in config_dict.items():
        setattr(a, k, v)
    a.n_threads = n_threads
    a.tree_method = "hist"
    a.multi_output = False
    return a


def evaluate_inline(real_12, gen_12):
    """Quick inline evaluation: AUC, correlation error, consistency."""
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

    # Consistency (feature order: d12,d2,mass,pt,tau1,tau2,tau3,tau21,tau32,ecf2,ecf3,d2_obs)
    c_tau21 = tau21_consistency_check(gen_12[:, 4], gen_12[:, 5], gen_12[:, 7])
    c_tau32 = tau32_consistency_check(gen_12[:, 5], gen_12[:, 6], gen_12[:, 8])

    print(f"\n{'='*60}")
    print(f"AUC (9 indep):   {r9['auc_test']:.4f}")
    print(f"AUC (12 feat):   {r12['auc_test']:.4f}")
    print(f"Mean |Δρ|:       {diff[mask_tri].mean():.4f}")
    print(f"Max  |Δρ|:       {diff[mask_tri].max():.4f}")
    print(f"Consistency tau21 W1: {c_tau21['w1']:.4f}")
    print(f"Consistency tau32 W1: {c_tau32['w1']:.4f}")
    print(f"{'='*60}\n")

    return {
        "auc_9": r9["auc_test"], "auc_12": r12["auc_test"],
        "mean_delta_rho": float(diff[mask_tri].mean()),
        "max_delta_rho": float(diff[mask_tri].max()),
        "consistency_tau21_w1": c_tau21["w1"],
        "consistency_tau32_w1": c_tau32["w1"],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = CONFIGS[args.config]
    seeds = [int(s) for s in args.ensemble_seeds.split(",")] if args.ensemble_seeds else [1980]

    print(f"Config: {args.config} = {config}")
    print(f"Strategies: residual={args.residual_stacking}, "
          f"ensemble={len(seeds)} seeds, cholesky={args.cholesky_9feat}")

    # Load and strip derived features (always)
    X_raw, y = load_data(args.data)
    print(f"Loaded {X_raw.shape}")
    X, derived_info = strip_derived_features(X_raw)
    print(f"Stripped to {X.shape[1]} independent features")

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    b, c = X.shape

    train_args = make_args_obj(config, n_threads=args.n_threads)
    n_t = args.n_timesteps

    # Train models for each seed
    all_models = []  # list of (pass1, pass2_or_None) per seed
    total_t0 = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        perm = np.random.permutation(b)
        X_shuf, y_shuf = X[perm], y[perm]

        X_scaled, scaler, mask_y, y_uniques, y_probs = prepare_scaling(
            X_shuf, y_shuf, args.duplicate_k
        )

        t0 = time.time()
        if args.residual_stacking:
            regr_p1, regr_p2 = train_batched_with_residuals(
                X_scaled, y_shuf, args.duplicate_k, n_t, "icfm", 0.0,
                mask_y, y_uniques, c, train_args,
            )
            all_models.append((regr_p1, regr_p2, y_uniques, y_probs, mask_y))
        else:
            regr = train_batched(
                X_scaled, y_shuf, args.duplicate_k, n_t, "icfm", 0.0,
                mask_y, y_uniques, c, train_args,
            )
            all_models.append((regr, None, y_uniques, y_probs, mask_y))
        print(f"Seed {seed} training: {time.time()-t0:.0f}s")

    print(f"\nTotal training time: {time.time()-total_t0:.0f}s")

    # Use the first seed's scaler for inverse transform (all should be similar)
    # Re-fit scaler on unshuffled data for consistency
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    perm = np.random.permutation(b)
    X_shuf = X[perm]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_shuf)

    # Sample
    n_samples = b
    x0 = np.random.normal(size=(n_samples, c))

    # Use first seed's class distribution
    y_uniques_0, y_probs_0 = all_models[0][3 - 1], all_models[0][4 - 1]  # y_probs, mask_y
    # Actually let's just use the stored values
    y_uniques = all_models[0][2]
    y_probs = all_models[0][3]

    label_y = y_uniques[np.argmax(
        np.random.multinomial(1, y_probs, size=n_samples), axis=1
    )]
    mask_y_fake = {label: (label_y == label) for label in y_uniques}

    # Build model functions
    model_fns = []
    for regr_p1, regr_p2, y_u, y_p, _ in all_models:
        if regr_p2 is not None:
            fn = build_model_fn_residual(regr_p1, regr_p2, y_u, c, n_t, mask_y_fake)
        else:
            fn = build_model_fn(regr_p1, y_u, c, n_t, mask_y_fake)
        model_fns.append(fn)

    if len(model_fns) > 1:
        model_fn = build_model_fn_ensemble(model_fns)
    else:
        model_fn = model_fns[0]

    print(f"Sampling {n_samples} events with Euler (N={args.solver_steps})...")
    t0 = time.time()
    solution = euler_solve(x0=x0.reshape(-1), my_model=model_fn, N=args.solver_steps)
    print(f"Sampling: {time.time()-t0:.1f}s")

    solution = solution.reshape(n_samples, c)
    solution = scaler.inverse_transform(solution)
    solution = np.clip(solution, X_min, X_max)

    # Strategy B: Cholesky on 9 independent features (blended)
    if args.cholesky_9feat:
        print(f"Applying Cholesky correction on 9 independent features (alpha={args.cholesky_alpha})...")
        solution = cholesky_correction(solution, X, alpha=args.cholesky_alpha)

    # Restore derived features
    solution_full = restore_derived_features(solution, derived_info)
    print(f"Restored to {solution_full.shape[1]} features")

    # Save
    np.save(os.path.join(args.output_dir, "generated_samples.npy"), solution_full)
    print(f"Saved to {args.output_dir}")

    # Save models
    with open(os.path.join(args.output_dir, "models.pkl"), "wb") as f:
        pickle.dump(all_models, f)

    # Evaluate
    metrics = evaluate_inline(X_raw, solution_full)

    # Save metrics
    import json
    with open(os.path.join(args.output_dir, "quick_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
