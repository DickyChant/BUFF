"""MLP-based conditional flow matching baseline for comparison with BUFF/flowBDT.

Trains a standard neural network on the same flow matching objective to demonstrate
that BDT achieves competitive or better results at similar compute budget.

Usage:
  python scripts/train_mlp_baseline.py --hidden 256,256,256 --epochs 200 --batch-size 4096
  python scripts/train_mlp_baseline.py --hidden 128,128 --epochs 100  # smaller/faster
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance

sys.path.insert(0, ".")
try:
    from BUFF.runner.train_and_sample import (
        strip_derived_features, restore_derived_features, JETNET_INDEPENDENT_INDICES,
    )
    from BUFF.runner.ode_example import euler_solve
    from BUFF.evaluation.discriminator import train_discriminator
    from BUFF.evaluation.consistency import tau21_consistency_check, tau32_consistency_check
except ImportError:
    from runner.train_and_sample import (
        strip_derived_features, restore_derived_features, JETNET_INDEPENDENT_INDICES,
    )
    from runner.ode_example import euler_solve
    from evaluation.discriminator import train_discriminator
    from evaluation.consistency import tau21_consistency_check, tau32_consistency_check


class FlowMLP(nn.Module):
    """MLP velocity field: (t, x_t) -> v_t."""

    def __init__(self, dim, hidden_sizes, time_embed_dim=32):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_sizes[0]),
            nn.SiLU(),
        )

        layers = []
        in_dim = dim + hidden_sizes[0]  # x_t concatenated with time embedding
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def time_embedding(self, t):
        """Sinusoidal positional embedding for timestep."""
        half = self.time_embed_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t, x):
        t_emb = self.time_embedding(t)
        t_feat = self.time_mlp(t_emb)
        h = torch.cat([x, t_feat], dim=-1)
        return self.net(h)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/jetnet/t_hlv12.npy")
    p.add_argument("--hidden", type=str, default="256,256,256",
                   help="Comma-separated hidden layer sizes")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Flow matching noise sigma")
    p.add_argument("--solver-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=1980)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", default="results/mlp_baseline/")
    p.add_argument("--max-train-time", type=float, default=9300,
                   help="Max training time in seconds (default: match BDT ~2.6h)")
    return p.parse_args()


def train_mlp_flow(model, X_scaled, epochs, batch_size, lr, weight_decay,
                   sigma, device, max_train_time):
    """Train MLP on flow matching objective.

    For each batch:
      1. Sample x1 from data, x0 ~ N(0,1), t ~ U(0,1)
      2. Compute x_t = (1-t)*x0 + t*x1 + sigma*eps  (conditional flow)
      3. True velocity u_t = x1 - x0
      4. Loss = ||model(t, x_t) - u_t||^2
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tensor = torch.from_numpy(X_scaled).float().to(device)
    n = X_tensor.shape[0]

    model.train()
    t0 = time.time()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n)
        X_shuffled = X_tensor[perm]

        for i in range(0, n, batch_size):
            x1 = X_shuffled[i:i+batch_size]
            bs = x1.shape[0]

            x0 = torch.randn_like(x1)
            t = torch.rand(bs, device=device)

            # Conditional flow: x_t = (1-t)*x0 + t*x1
            t_expand = t[:, None]
            x_t = (1 - t_expand) * x0 + t_expand * x1
            if sigma > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

            # True velocity: u_t = x1 - x0
            u_t = x1 - x0

            # Predict and compute loss
            v_pred = model(t, x_t)
            loss = ((v_pred - u_t) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        elapsed = time.time() - t0
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s")

        # Time budget check
        if elapsed > max_train_time:
            print(f"Time budget reached ({elapsed:.0f}s > {max_train_time:.0f}s), stopping at epoch {epoch+1}")
            break

    total_time = time.time() - t0
    print(f"Training done: {epoch+1} epochs in {total_time:.0f}s, final loss={losses[-1]:.6f}")
    return losses, total_time


@torch.no_grad()
def sample_mlp(model, n_samples, dim, solver_steps, device):
    """Sample from trained MLP flow using Euler solver."""
    model.eval()
    x = torch.randn(n_samples, dim, device=device)
    h = 1.0 / solver_steps

    for i in range(solver_steps):
        t = torch.ones(n_samples, device=device) * (i * h)
        v = model(t, x)
        x = x + h * v

    return x.cpu().numpy()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hidden_sizes = [int(x) for x in args.hidden.split(",")]
    device = args.device

    # Load data
    X_raw = np.load(args.data)
    print(f"Loaded {X_raw.shape}")

    # Strip derived features (same as BDT pipeline)
    X, derived_info = strip_derived_features(X_raw)
    print(f"Stripped to {X.shape[1]} independent features")

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    b, c = X.shape

    perm = np.random.permutation(b)
    X = X[perm]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Build model
    model = FlowMLP(dim=c, hidden_sizes=hidden_sizes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: {hidden_sizes}, {n_params:,} parameters")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Time budget: {args.max_train_time:.0f}s")

    # Train
    losses, train_time = train_mlp_flow(
        model, X_scaled, args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.sigma, device, args.max_train_time,
    )

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "hidden_sizes": hidden_sizes,
        "dim": c,
        "losses": losses,
        "train_time": train_time,
    }, os.path.join(args.output_dir, "model.pt"))

    # Sample
    print(f"\nSampling {b} events with Euler ({args.solver_steps} steps)...")
    t0 = time.time()
    solution = sample_mlp(model, b, c, args.solver_steps, device)
    sample_time = time.time() - t0
    print(f"Sampling: {sample_time:.1f}s")

    solution = scaler.inverse_transform(solution)
    solution = np.clip(solution, X_min, X_max)

    # Restore derived features
    full = np.zeros((b, 12))
    for ni, oi in enumerate(JETNET_INDEPENDENT_INDICES):
        full[:, oi] = solution[:, ni]
    full[:, 7] = full[:, 5] / np.clip(full[:, 4], 1e-8, None)
    full[:, 8] = full[:, 6] / np.clip(full[:, 5], 1e-8, None)
    full[:, 11] = full[:, 10] / np.clip(full[:, 9]**2, 1e-16, None)

    np.save(os.path.join(args.output_dir, "generated_samples.npy"), full)

    # Evaluate
    print("\n=== Evaluation ===")
    X_raw_orig = np.load(args.data)

    r9 = train_discriminator(
        X_raw_orig[:, JETNET_INDEPENDENT_INDICES],
        full[:, JETNET_INDEPENDENT_INDICES],
    )
    r12 = train_discriminator(X_raw_orig, full)

    corr_diff = np.abs(np.corrcoef(X_raw_orig.T) - np.corrcoef(full.T))
    np.fill_diagonal(corr_diff, 0)
    mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)

    c_tau21 = tau21_consistency_check(full[:, 4], full[:, 5], full[:, 7])
    c_tau32 = tau32_consistency_check(full[:, 5], full[:, 6], full[:, 8])

    features = ["d12", "d2", "mass", "pt", "tau1", "tau2", "tau3",
                "tau21", "tau32", "ecf2", "ecf3", "d2_obs"]
    print(f"\nPer-feature W1:")
    w1s = {}
    for i, f in enumerate(features):
        w = wasserstein_distance(X_raw_orig[:, i], full[:, i])
        w1s[f] = w
        print(f"  {f:>8}: {w:.6f}")

    print(f"\n{'='*60}")
    print(f"MLP: {hidden_sizes}, {n_params:,} params, {train_time:.0f}s training")
    print(f"AUC (9 indep):   {r9['auc_test']:.4f}")
    print(f"AUC (12 feat):   {r12['auc_test']:.4f}")
    print(f"Mean |Δρ|:       {corr_diff[mask].mean():.4f}")
    print(f"Max  |Δρ|:       {corr_diff[mask].max():.4f}")
    print(f"Consistency tau21 W1: {c_tau21['w1']:.4f}")
    print(f"Consistency tau32 W1: {c_tau32['w1']:.4f}")
    print(f"Sampling time:   {sample_time:.1f}s")
    print(f"{'='*60}")

    # Save metrics
    metrics = {
        "model": f"MLP {hidden_sizes}",
        "n_params": n_params,
        "train_time_s": train_time,
        "sample_time_s": sample_time,
        "epochs_completed": len(losses),
        "final_loss": losses[-1],
        "auc_9": r9["auc_test"],
        "auc_12": r12["auc_test"],
        "mean_delta_rho": float(corr_diff[mask].mean()),
        "max_delta_rho": float(corr_diff[mask].max()),
        "consistency_tau21_w1": c_tau21["w1"],
        "consistency_tau32_w1": c_tau32["w1"],
        "w1_per_feature": {f: float(w) for f, w in w1s.items()},
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
