"""Evaluate CaloChallenge HLV features (flowBDT vs real).

Computes:
- Discriminator AUC (MLP) on high-level features
- Wasserstein-1 distances on individual HLV features
- Bootstrap uncertainties for all metrics
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from BUFF.evaluation.wasserstein import w1_with_errors
from BUFF.evaluation.metrics import sep_power_with_errors
from BUFF.evaluation.discriminator import train_discriminator, discriminator_auc_with_errors


FEATURE_NAMES = [
    "response", "layer0_frac", "layer1_frac", "layer2_frac",
    "layer3_frac", "layer4_frac", "depth", "width",
]


def main():
    output_dir = "rebuttal/results/calo/"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    real_hlv = np.load("data/calo/processed/calo_hlv.npy")
    gen_hlv = np.load("results/calo_hlv/generated_samples.npy")
    print(f"Real HLV: {real_hlv.shape}, Gen HLV: {gen_hlv.shape}")

    results = {}

    # 1. Discriminator on HLV features
    print("Training discriminator on HLV features...")
    disc_single = train_discriminator(real_hlv, gen_hlv, seed=42)
    results["discriminator_single"] = disc_single
    print(f"  AUC (test) = {disc_single['auc_test']:.4f}")

    print("Bootstrap discriminator (20 iterations)...")
    disc_boot = discriminator_auc_with_errors(real_hlv, gen_hlv, n_bootstrap=20, seed=42)
    results["discriminator_bootstrap"] = disc_boot
    print(f"  AUC = {disc_boot['auc_mean']:.4f} +/- {disc_boot['auc_std']:.4f}")

    # ROC curve
    X = np.vstack([real_hlv, gen_hlv])
    y = np.concatenate([np.ones(len(real_hlv)), np.zeros(len(gen_hlv))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {disc_single['auc_test']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Discriminator ROC - CaloChallenge HLV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.pdf"), dpi=150)
    plt.close(fig)

    # 2. W1 distances per HLV feature
    print("\nComputing W1 distances per HLV feature...")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i+1}/{len(FEATURE_NAMES)}] {name}")
        w1 = w1_with_errors(real_hlv[:, i], gen_hlv[:, i], n_bootstrap=100)
        sep = sep_power_with_errors(real_hlv[:, i], gen_hlv[:, i], n_bootstrap=100)
        results[f"hlv_{name}"] = {
            "w1_mean": w1["w1_mean"], "w1_std": w1["w1_std"],
            "w1_truth_mean": w1["truth_mean"], "w1_truth_std": w1["truth_std"],
            "sep_mean": sep["sep_mean"], "sep_std": sep["sep_std"],
            "sep_truth_mean": sep["truth_mean"], "sep_truth_std": sep["truth_std"],
        }
        print(f"    W1 = {w1['w1_mean']:.6f} +/- {w1['w1_std']:.6f} "
              f"(truth: {w1['truth_mean']:.6f} +/- {w1['truth_std']:.6f})")

    # Save
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
