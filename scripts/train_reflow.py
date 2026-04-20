"""Reflow-style distillation for 1-step BUFF sampling.

This script distills a trained BUFF teacher into a time-independent
multi-output XGBoost student. The student is trained on teacher-generated
pairs (x0, x1_teacher) in the scaled BUFF model space, so a single Euler
update with h=1 can land near the teacher endpoint.

Typical usage:
  uv run python scripts/train_reflow.py \
      --teacher-dir results/jetnet_highlevel_v3 \
      --data data/jetnet/t_hlv12.npy \
      --output-dir results/reflow_1step/
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
try:
    from BUFF.runner.train_and_sample import (
        JETNET_INDEPENDENT_INDICES,
        build_model_fn,
        load_data,
        restore_derived_features,
        sample_one_step,
        train_reflow_student,
    )
    from BUFF.runner.ode_example import (
        dpori5_solve_numpy, euler_solve, midpoint_solve,
    )
    from BUFF.evaluation.consistency import (
        tau21_consistency_check, tau32_consistency_check,
    )
    from BUFF.evaluation.discriminator import train_discriminator
except ImportError:
    from runner.train_and_sample import (
        JETNET_INDEPENDENT_INDICES,
        build_model_fn,
        load_data,
        restore_derived_features,
        sample_one_step,
        train_reflow_student,
    )
    from runner.ode_example import (
        dpori5_solve_numpy, euler_solve, midpoint_solve,
    )
    from evaluation.consistency import (
        tau21_consistency_check, tau32_consistency_check,
    )
    from evaluation.discriminator import train_discriminator


class ArgsAdapter:
    """Adapter to pass CLI args to shared XGBoost helpers."""
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Distill a BUFF teacher into a 1-step student")
    p.add_argument(
        "--teacher-dir", required=True,
        help="Directory containing teacher models.pkl and scaler.pkl",
    )
    p.add_argument(
        "--data", default="data/jetnet/t_hlv12.npy",
        help="Optional real data for pair count and evaluation",
    )
    p.add_argument(
        "--n-pairs", type=int, default=0,
        help="Number of teacher-generated pairs (0 = match real-data size)",
    )
    p.add_argument(
        "--teacher-solver",
        choices=["euler", "midpoint", "dopri5"],
        default="dopri5",
        help="Solver used to generate teacher endpoints for distillation",
    )
    p.add_argument(
        "--teacher-solver-steps", type=int, default=0,
        help="Teacher solve steps (0 = use teacher n_t)",
    )
    p.add_argument("--seed", type=int, default=1980)
    p.add_argument("--n-threads", type=int, default=16)

    # Student XGBoost hyperparameters
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.5)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9, dest="colsample_bytree")
    p.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="XGBoost device for the student",
    )

    p.add_argument(
        "--feature-interactions", action="store_true",
        help="Add pairwise x0 feature products to the student input",
    )
    p.add_argument(
        "--predict-endpoint", action="store_true",
        help="Train the student to predict x1 directly instead of displacement",
    )
    p.add_argument("--output-dir", default="results/reflow_1step/")
    return p.parse_args()


def make_train_args(args):
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
    return train_a


def load_teacher(teacher_dir):
    models_path = os.path.join(teacher_dir, "models.pkl")
    meta_path = os.path.join(teacher_dir, "scaler.pkl")

    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Missing teacher models: {models_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing teacher metadata: {meta_path}")

    with open(models_path, "rb") as f:
        teacher_models = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    required_keys = ["scaler", "X_min", "X_max", "y_uniques", "y_probs", "c", "n_t"]
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise KeyError(f"Teacher metadata missing keys: {missing}")

    return teacher_models, meta


def generate_teacher_pairs(teacher_models, meta, n_pairs, solver_name, solver_steps):
    c = meta["c"]
    y_uniques = meta["y_uniques"]
    y_probs = meta["y_probs"]

    x0 = np.random.normal(size=(n_pairs, c))
    label_y = y_uniques[np.argmax(
        np.random.multinomial(1, y_probs, size=n_pairs), axis=1
    )]
    mask_y = {label: (label_y == label) for label in y_uniques}

    teacher_fn = build_model_fn(
        teacher_models, y_uniques, c, meta["n_t"], mask_y,
        multi_output=meta.get("multi_output", False),
    )

    solvers = {
        "euler": euler_solve,
        "midpoint": midpoint_solve,
        "dopri5": dpori5_solve_numpy,
    }
    solve = solvers[solver_name]

    print(f"Generating {n_pairs} teacher pairs with {solver_name} ({solver_steps} steps)...")
    t0 = time.time()
    x1_teacher = solve(x0=x0.reshape(-1), my_model=teacher_fn, N=solver_steps)
    elapsed = time.time() - t0
    print(f"Teacher pair generation: {elapsed:.1f}s")

    x1_teacher = x1_teacher.reshape(n_pairs, c)
    x1_teacher = np.clip(x1_teacher, -1.0, 1.0)

    return x0, x1_teacher, label_y


def evaluate(real, gen):
    from scipy.stats import wasserstein_distance

    metrics = {}

    if real.shape[1] == 12 and gen.shape[1] == 12:
        r9 = train_discriminator(
            real[:, JETNET_INDEPENDENT_INDICES],
            gen[:, JETNET_INDEPENDENT_INDICES],
        )
        metrics["auc_9"] = r9["auc_test"]

    r_all = train_discriminator(real, gen)
    metrics["auc_all"] = r_all["auc_test"]

    corr_real = np.corrcoef(real.T)
    corr_gen = np.corrcoef(gen.T)
    diff = np.abs(corr_real - corr_gen)
    np.fill_diagonal(diff, 0)
    mask_tri = np.triu(np.ones_like(diff, dtype=bool), k=1)
    metrics["mean_delta_rho"] = float(diff[mask_tri].mean())
    metrics["max_delta_rho"] = float(diff[mask_tri].max())

    feature_names = [f"f{i}" for i in range(real.shape[1])]
    if real.shape[1] == 12:
        feature_names = [
            "d12", "d2", "mass", "pt", "tau1", "tau2", "tau3",
            "tau21", "tau32", "ecf2", "ecf3", "d2_obs",
        ]
        c_tau21 = tau21_consistency_check(gen[:, 4], gen[:, 5], gen[:, 7])
        c_tau32 = tau32_consistency_check(gen[:, 5], gen[:, 6], gen[:, 8])
        metrics["consistency_tau21_w1"] = c_tau21["w1"]
        metrics["consistency_tau32_w1"] = c_tau32["w1"]

    w1s = {}
    for i, name in enumerate(feature_names):
        w1s[name] = float(wasserstein_distance(real[:, i], gen[:, i]))
    metrics["w1_per_feature"] = w1s

    print(f"\n{'='*60}")
    if "auc_9" in metrics:
        print(f"AUC (9 indep):   {metrics['auc_9']:.4f}")
    print(f"AUC (all feat):  {metrics['auc_all']:.4f}")
    print(f"Mean |Δρ|:       {metrics['mean_delta_rho']:.4f}")
    print(f"Max  |Δρ|:       {metrics['max_delta_rho']:.4f}")
    if "consistency_tau21_w1" in metrics:
        print(f"Consistency tau21 W1: {metrics['consistency_tau21_w1']:.4f}")
        print(f"Consistency tau32 W1: {metrics['consistency_tau32_w1']:.4f}")
    print(f"{'='*60}\n")

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    teacher_models, meta = load_teacher(args.teacher_dir)

    X_real = None
    if args.data and os.path.exists(args.data):
        X_real, _ = load_data(args.data)
        print(f"Loaded real data {X_real.shape} for evaluation/reference")
    elif args.n_pairs <= 0:
        raise ValueError("--n-pairs must be > 0 when --data is not provided")

    n_pairs = args.n_pairs if args.n_pairs > 0 else X_real.shape[0]
    solver_steps = args.teacher_solver_steps if args.teacher_solver_steps > 0 else meta["n_t"]

    x0, x1_teacher, label_y = generate_teacher_pairs(
        teacher_models, meta, n_pairs, args.teacher_solver, solver_steps
    )

    train_args = make_train_args(args)

    print("\n=== Distilling 1-step student ===")
    print(f"  Hyperparameters: depth={args.max_depth}, n_est={args.n_estimators}, "
          f"eta={args.eta}, interactions={args.feature_interactions}, "
          f"predict_endpoint={args.predict_endpoint}")
    t0 = time.time()
    student_models, input_dim = train_reflow_student(
        x0, x1_teacher, label_y, meta["y_uniques"], train_args,
        predict_endpoint=args.predict_endpoint,
        feature_interactions=args.feature_interactions,
    )
    train_time_s = time.time() - t0
    print(f"Student training: {train_time_s:.1f}s")

    with open(os.path.join(args.output_dir, "models.pkl"), "wb") as f:
        pickle.dump(student_models, f)
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump({
            "scaler": meta["scaler"],
            "X_min": meta["X_min"],
            "X_max": meta["X_max"],
            "y_uniques": meta["y_uniques"],
            "y_probs": meta["y_probs"],
            "c": meta["c"],
            "n_t": 2,
            "multi_output": True,
            "derived_info": meta.get("derived_info"),
            "student_type": "reflow_one_step",
            "student_input_dim": input_dim,
            "predict_endpoint": args.predict_endpoint,
            "feature_interactions": args.feature_interactions,
            "teacher_dir": args.teacher_dir,
            "teacher_solver": args.teacher_solver,
            "teacher_solver_steps": solver_steps,
            "n_pairs": n_pairs,
        }, f)

    t0 = time.time()
    generated, labels = sample_one_step(
        student_models, meta["y_uniques"], meta["y_probs"], meta["c"],
        meta["scaler"], meta["X_min"], meta["X_max"], n_pairs,
        predict_endpoint=args.predict_endpoint,
        feature_interactions=args.feature_interactions,
    )
    sample_time_s = time.time() - t0

    if meta.get("derived_info") is not None:
        generated_full = restore_derived_features(generated, meta["derived_info"])
    else:
        generated_full = generated

    np.save(os.path.join(args.output_dir, "generated_samples.npy"), generated_full)
    np.save(os.path.join(args.output_dir, "generated_labels.npy"), labels)

    metrics = {
        "method": "reflow_one_step",
        "train_time_s": train_time_s,
        "sample_time_s": sample_time_s,
        "n_pairs": n_pairs,
        "teacher_solver": args.teacher_solver,
        "teacher_solver_steps": solver_steps,
        "student_hyperparameters": {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "eta": args.eta,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "feature_interactions": args.feature_interactions,
            "predict_endpoint": args.predict_endpoint,
        },
    }

    if X_real is not None:
        if X_real.shape[1] == generated_full.shape[1]:
            metrics.update(evaluate(X_real, generated_full))
        else:
            print("Skipping evaluation: real/generated feature counts do not match")
            metrics["evaluation_skipped"] = "shape_mismatch"

    with open(os.path.join(args.output_dir, "quick_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved distilled student to {args.output_dir}")


if __name__ == "__main__":
    main()
