"""Pair-feature discriminator AUC for JetNet HLV samples.

Trains a 2-D MLP discriminator on every pair (i, j) of the 12 HLV features,
to localize WHERE the real-vs-gen discriminator picks up signal.  Outputs a
12x12 AUC matrix and a sorted list of the most/least distinguishable pairs.

Use this to answer: 'is the joint-correlation gap concentrated in a few
specific feature pairs (interpretable physics), or spread evenly across
all pairs (a structural bottleneck)?'

Usage
-----
    python -m BUFF.scripts.run_jetnet_hlv_pair_auc \
        --real /path/to/t_hlv12.npy \
        --gen  /path/to/generated_samples.npy \
        --out-dir rebuttal/results/jetnet_hlv_pairs/ \
        --n-events 30000
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


FEATURE_LABELS = ["d_{12}", "d_{23}", "mass", "p_T",
                  "tau_1", "tau_2", "tau_3",
                  "tau_{21}", "tau_{32}",
                  "ecf_2", "ecf_3", "d_2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-events", type=int, default=30000,
                   help="Subsample size per side. Smaller = faster but noisier.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    real = np.load(args.real).astype(np.float32)
    gen = np.load(args.gen).astype(np.float32)
    d = real.shape[1]
    if gen.shape[1] != d:
        raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")

    n = min(len(real), len(gen), args.n_events)
    real_s = real[rng.choice(len(real), n, replace=False)]
    gen_s = gen[rng.choice(len(gen), n, replace=False)]
    print(f"[pair-auc] {d} features, {d*(d-1)//2} pairs, {n} events/side")

    # Single-feature AUC (along the diagonal of the matrix)
    print("[single] per-feature AUC:")
    single = {}
    for i in range(d):
        r = train_discriminator(real_s[:, i:i+1], gen_s[:, i:i+1], seed=args.seed)
        single[FEATURE_LABELS[i]] = float(r["auc_test"])
        print(f"  {FEATURE_LABELS[i]:10s}  AUC = {r['auc_test']:.4f}")

    # Pair AUC matrix
    print("[pair] computing pair AUCs (this is the slow part)...")
    auc_mat = np.eye(d, dtype=np.float64) * 0.5  # diagonal = single-feat (placeholder)
    pair_dict = {}
    for i in range(d):
        for j in range(i + 1, d):
            r = train_discriminator(real_s[:, [i, j]], gen_s[:, [i, j]], seed=args.seed)
            v = float(r["auc_test"])
            auc_mat[i, j] = v
            auc_mat[j, i] = v
            key = f"{FEATURE_LABELS[i]} x {FEATURE_LABELS[j]}"
            pair_dict[key] = v
        # diagonal = single-feature for plotting convenience
        auc_mat[i, i] = single[FEATURE_LABELS[i]]
        print(f"  row {i+1}/{d} done")

    # Sort and report
    sorted_pairs = sorted(pair_dict.items(), key=lambda kv: -kv[1])
    print("\n[top 10] most distinguishable pairs:")
    for k, v in sorted_pairs[:10]:
        print(f"  {k:35s} AUC = {v:.4f}")
    print("\n[bottom 10] most fully captured pairs:")
    for k, v in sorted_pairs[-10:]:
        print(f"  {k:35s} AUC = {v:.4f}")

    out = {
        "config": vars(args),
        "n_used": int(n),
        "feature_labels": FEATURE_LABELS,
        "single_feature_auc": single,
        "pair_auc": pair_dict,
        "auc_matrix": auc_mat.tolist(),
        "top_10_pairs": sorted_pairs[:10],
        "bottom_10_pairs": sorted_pairs[-10:],
    }
    json_path = os.path.join(args.out_dir, "pair_auc.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(args.out_dir, "pair_auc_matrix.npy"), auc_mat)
    print(f"\n[save] {json_path}")
    print(f"[save] {args.out_dir}/pair_auc_matrix.npy")

    # Optional heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        # Use raw matplotlib labels (no LaTeX) to avoid font rendering issues
        plain_labels = ["d12","d23","mass","pT","tau1","tau2","tau3","tau21","tau32","ecf2","ecf3","d2"]
        im = ax.imshow(auc_mat, cmap="viridis", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(d)); ax.set_yticks(range(d))
        ax.set_xticklabels(plain_labels, rotation=45, ha="right")
        ax.set_yticklabels(plain_labels)
        for i in range(d):
            for j in range(d):
                ax.text(j, i, f"{auc_mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if auc_mat[i, j] > 0.75 else "black", fontsize=7)
        ax.set_title("Pair-feature discriminator AUC\n(diagonal = single-feature; off-diag = 2-D MLP)")
        plt.colorbar(im, ax=ax, label="AUC")
        plt.tight_layout()
        heatmap_path = os.path.join(args.out_dir, "pair_auc_heatmap.pdf")
        plt.savefig(heatmap_path, dpi=200)
        plt.close(fig)
        print(f"[save] {heatmap_path}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
