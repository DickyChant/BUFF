"""NN-flow baseline for the JetNet HLV discriminator-AUC comparison.

Mirrors the BUFF (BDT-flowBDT) pipeline on the same dataset and metric so
the two backbones can be compared apples-to-apples:

    BUFF (multi-output BDT, depth=6, n_est=200, eta=0.05) -- AUC ~= 0.997
    NN-CFM (MLP, hidden=128, depth=4, ~30 epochs)         -- THIS SCRIPT

The training objective is identical (MSE on the conditional flow-matching
velocity field, ICFM with sigma=0); only the regressor backbone differs.
This isolates the *expressivity* contribution of NN vs BDT for closing
the joint-correlation gap that the discriminator detects.

Usage
-----
    python -m BUFF.scripts.run_neural_flow_auc \
        --data /path/to/t_hlv12.npy \
        --out-dir rebuttal/results/jetnet_hlv_nn/ \
        --epochs 30 --hidden 128 --depth 4 --n-steps 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from BUFF.evaluation.discriminator import train_discriminator
from BUFF.scripts.eval_neural_flow_cpu import MLPVelocity, sample_euler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="JetNet HLV npy (N, 12)")
    p.add_argument("--out-dir", default="rebuttal/results/jetnet_hlv_nn/")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time-emb-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=0, help="0 = use all events for training")
    p.add_argument("--n-steps", type=int, default=30, help="ODE Euler steps for sampling")
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-threads", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.n_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load + scale ---
    print(f"[load] {args.data}")
    X_raw = np.load(args.data).astype(np.float32)
    print(f"  shape: {X_raw.shape}")
    n_total = len(X_raw)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    X_raw = X_raw[perm]

    n_train = args.n_train if args.n_train > 0 else n_total
    n_train = min(n_train, n_total)
    X_train_raw = X_raw[:n_train]
    print(f"  training on {n_train} events")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_train_raw)
    d = X_scaled.shape[1]

    # --- Train ICFM-MLP ---
    device = torch.device("cpu")
    model = MLPVelocity(d, hidden=args.hidden, depth=args.depth,
                         time_emb_dim=args.time_emb_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MLPVelocity hidden={args.hidden} depth={args.depth} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    fm = ConditionalFlowMatcher(sigma=0.0)
    X_torch = torch.from_numpy(X_scaled)

    n_iters = args.epochs * max(1, n_train // args.batch_size)
    print(f"[train] {args.epochs} epochs × {max(1, n_train // args.batch_size)} steps "
          f"(batch {args.batch_size}, total {n_iters} grad steps)")
    t0 = time.time()
    last_loss = None
    for it in range(n_iters):
        idx = torch.randint(0, n_train, (args.batch_size,))
        x1 = X_torch[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0])
        _, xt, ut = fm.sample_location_and_conditional_flow(x0, x1, t=t)
        v = model(xt, t)
        loss = ((v - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if (it + 1) % max(1, n_iters // 10) == 0:
            print(f"    iter {it+1:5d}/{n_iters}  loss={last_loss:.5f}")
    train_time = time.time() - t0
    print(f"[train] done in {train_time:.1f}s  ({train_time/n_iters*1000:.1f} ms/iter)")

    # --- Sample N = same as training set ---
    print(f"[sample] {n_train} events, {args.n_steps} Euler steps")
    t0 = time.time()
    gen_scaled = sample_euler(model, n_train, args.n_steps).cpu().numpy()
    sample_time = time.time() - t0
    print(f"  sampling: {sample_time:.1f}s  ({sample_time/n_train*1000:.3f} ms/event)")
    gen = scaler.inverse_transform(gen_scaled)
    # Clip to training data range (mirrors BUFF's np.clip step)
    gen = np.clip(gen, X_train_raw.min(axis=0), X_train_raw.max(axis=0))

    # --- Discriminator AUC bootstrap ---
    print(f"[discriminator] bootstrap n={args.n_bootstrap}")
    real = X_train_raw  # same N as gen
    rng = np.random.RandomState(args.seed)
    aucs = []
    for b in range(args.n_bootstrap):
        idx_r = rng.randint(0, len(real), size=len(real))
        idx_g = rng.randint(0, len(gen), size=len(gen))
        r = train_discriminator(real[idx_r], gen[idx_g], seed=args.seed + b)
        aucs.append(r["auc_test"])
        print(f"  bootstrap {b+1}/{args.n_bootstrap}: AUC = {r['auc_test']:.4f}")
    aucs = np.array(aucs)

    # --- Per-feature AUC ---
    features = ["d_{12}", "d_{23}", "mass", "p_T", "tau_1", "tau_2", "tau_3",
                "tau_{21}", "tau_{32}", "ecf_2", "ecf_3", "d_2"]
    per_feature = {}
    print("[per-feature AUC]")
    for i in range(d):
        r = train_discriminator(real[:, i:i+1], gen[:, i:i+1], seed=args.seed)
        per_feature[features[i] if i < len(features) else f"f{i}"] = float(r["auc_test"])
        print(f"  {features[i] if i < len(features) else f'f{i}':10s}  AUC = {r['auc_test']:.4f}")

    # --- Boundary diagnostic ---
    X_min = real.min(axis=0)
    X_max = real.max(axis=0)
    at_b = np.zeros(len(gen), dtype=bool)
    for k in range(d):
        at_b |= (gen[:, k] == X_min[k]) | (gen[:, k] == X_max[k])
    boundary_frac = float(at_b.mean())
    print(f"[diag] boundary-clipped fraction: {boundary_frac*100:.2f}%")

    # --- Save ---
    out = {
        "config": {
            "data": args.data,
            "n_train": n_train,
            "hidden": args.hidden,
            "depth": args.depth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_steps": args.n_steps,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "n_params": int(n_params),
        },
        "timing": {
            "train_s": train_time,
            "sample_s": sample_time,
            "sample_ms_per_event": sample_time / n_train * 1000,
        },
        "auc": {
            "mean": float(aucs.mean()),
            "std": float(aucs.std()),
            "values": aucs.tolist(),
        },
        "boundary_fraction": boundary_frac,
        "per_feature_auc": per_feature,
        "last_loss": last_loss,
    }
    with open(os.path.join(args.out_dir, "nn_flow_auc.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(args.out_dir, "nn_generated_samples.npy"), gen.astype(np.float32))
    print(f"\n[save] {args.out_dir}/nn_flow_auc.json")
    print(f"[save] {args.out_dir}/nn_generated_samples.npy")
    print()
    print(f"NN-flow AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}   "
          f"(BUFF-BDT baseline = 0.9994; BUFF-BDT hicap+strip = 0.9974)")


if __name__ == "__main__":
    main()
