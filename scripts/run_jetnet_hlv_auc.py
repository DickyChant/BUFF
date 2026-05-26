"""Bootstrap AUC + per-feature AUC breakdown for JetNet HLV 12-feature samples.

Usage
-----
    python -m BUFF.scripts.run_jetnet_hlv_auc \
        --real /path/to/t_hlv12.npy \
        --gen  /path/to/generated_samples.npy \
        --out-dir rebuttal/results/jetnet_v3_hicap/ \
        --n-bootstrap 20

Outputs (in --out-dir):
    auc_bootstrap.json   bootstrap mean/std + per-feature AUC + diagnostics
    auc_table.tex        compact LaTeX summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from BUFF.evaluation.discriminator import train_discriminator


FEATURES_12 = [
    "d_{12}", "d_{23}", "mass", "p_T",
    "tau_1", "tau_2", "tau_3",
    "tau_{21}", "tau_{32}",
    "ecf_2", "ecf_3", "d_2",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--out-dir", default="rebuttal/results/jetnet_hlv_auc/")
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--max-events", type=int, default=0, help="0 = use min of real/gen")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def boundary_fraction(real, gen):
    X_min = real.min(axis=0)
    X_max = real.max(axis=0)
    at_b = np.zeros(len(gen), dtype=bool)
    for k in range(gen.shape[1]):
        at_b |= (gen[:, k] == X_min[k]) | (gen[:, k] == X_max[k])
    return float(at_b.mean()), at_b


def bootstrap_auc(real, gen, n_bootstrap, seed):
    rng = np.random.RandomState(seed)
    aucs = []
    for b in range(n_bootstrap):
        idx_r = rng.randint(0, len(real), size=len(real))
        idx_g = rng.randint(0, len(gen), size=len(gen))
        r = train_discriminator(real[idx_r], gen[idx_g], seed=seed + b)
        aucs.append(r["auc_test"])
        print(f"  bootstrap {b+1}/{n_bootstrap}: AUC_test = {r['auc_test']:.4f}")
    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std()), aucs.tolist()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] real: {args.real}")
    real = np.load(args.real).astype(np.float32)
    print(f"[load] gen:  {args.gen}")
    gen = np.load(args.gen).astype(np.float32)
    if real.shape[1] != gen.shape[1]:
        raise ValueError(f"feature mismatch: real {real.shape} vs gen {gen.shape}")
    n = args.max_events if args.max_events > 0 else min(len(real), len(gen))
    rng = np.random.RandomState(args.seed)
    real = real[rng.permutation(len(real))[:n]]
    gen = gen[rng.permutation(len(gen))[:n]]
    print(f"  using {n} events/side, {real.shape[1]} features")

    # 1) Bootstrap AUC on all features
    print(f"\n[bootstrap] full {real.shape[1]}-feature classifier, n={args.n_bootstrap}")
    auc_mean, auc_std, auc_values = bootstrap_auc(real, gen, args.n_bootstrap, args.seed)
    print(f"  AUC = {auc_mean:.4f} +/- {auc_std:.4f}")

    # 2) Boundary-clipping diagnostic
    frac, at_b = boundary_fraction(real, gen)
    print(f"\n[diag] gen events at any feature boundary: {frac*100:.2f}%")

    # 3) Per-feature single AUC (which feature is most distinguishable?)
    print("\n[diag] per-feature AUC (single-feature discriminator):")
    per_feature = {}
    for i in range(real.shape[1]):
        r = train_discriminator(real[:, i:i+1], gen[:, i:i+1], seed=args.seed)
        per_feature[FEATURES_12[i] if i < len(FEATURES_12) else f"f{i}"] = float(r["auc_test"])
        print(f"  {FEATURES_12[i] if i < len(FEATURES_12) else f'f{i}':12s}  AUC = {r['auc_test']:.4f}")

    out = {
        "config": {
            "real": args.real,
            "gen": args.gen,
            "n_events": n,
            "n_features": int(real.shape[1]),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "auc": {
            "mean": auc_mean,
            "std": auc_std,
            "values": auc_values,
        },
        "boundary_fraction": frac,
        "per_feature_auc": per_feature,
    }
    with open(os.path.join(args.out_dir, "auc_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Compact LaTeX line
    with open(os.path.join(args.out_dir, "auc_table.tex"), "w") as f:
        f.write(f"\\textbf{{AUC}} $= {auc_mean:.4f} \\pm {auc_std:.4f}$ "
                f"(bootstrap, $n={args.n_bootstrap}$; "
                f"boundary-clipping fraction = ${frac*100:.1f}\\%$).\n")

    print(f"\n[save] {args.out_dir}/auc_bootstrap.json")
    print(f"[save] {args.out_dir}/auc_table.tex")


if __name__ == "__main__":
    main()
