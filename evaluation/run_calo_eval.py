"""Full CaloChallenge evaluation with discriminator + W1 metrics.

Usage
-----
uv run python -m BUFF.evaluation.run_calo_eval \
    --real real_showers.npy --gen generated_showers.npy \
    --output-dir results/calo_eval/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from BUFF.evaluation.wasserstein import w1_with_errors
from BUFF.evaluation.discriminator import train_discriminator, discriminator_auc_with_errors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="Path to real showers .npy (N, 368)")
    p.add_argument("--gen", required=True, help="Path to generated showers .npy (N, 368)")
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="results/calo_eval/")
    return p.parse_args()


def compute_shower_response(showers):
    """Shower response: sum of all voxel energies per event."""
    return showers.sum(axis=1)


def compute_layer_energies(showers, layer_boundaries):
    """Sum voxel energies per calorimeter layer."""
    layers = {}
    for i, (start, end) in enumerate(layer_boundaries):
        layers[f"layer_{i}"] = showers[:, start:end].sum(axis=1)
    return layers


def compute_center_of_energy(showers, n_voxels):
    """Center of energy index (simplified): weighted mean of voxel index."""
    indices = np.arange(n_voxels)
    total = showers.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return (showers * indices[None, :]).sum(axis=1) / total.ravel()


# CaloChallenge dataset 1 layer boundaries (approximate, 368 voxels total)
DATASET1_LAYER_BOUNDARIES = [
    (0, 288),    # Layer 0+1+2 (fine granularity)
    (288, 332),  # Layer 3
    (332, 368),  # Layer 4
]


def plot_roc(real, gen, output_path, seed=42):
    """Train discriminator and plot ROC curve."""
    result = train_discriminator(real, gen, seed=seed)
    # Re-train for ROC data
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([real, gen])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(gen))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        random_state=seed, early_stopping=True)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {result['auc_test']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Discriminator ROC — CaloChallenge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real = np.load(args.real)
    gen = np.load(args.gen)
    print(f"Real: {real.shape}, Gen: {gen.shape}")

    results = {}

    # 1. Discriminator
    print("Training discriminator...")
    disc_single = plot_roc(real, gen, os.path.join(args.output_dir, "roc_curve.pdf"))
    results["discriminator_single"] = disc_single
    print(f"  AUC (test) = {disc_single['auc_test']:.4f}")

    print(f"Bootstrap discriminator ({args.n_bootstrap} iterations)...")
    disc_boot = discriminator_auc_with_errors(real, gen, n_bootstrap=args.n_bootstrap)
    results["discriminator_bootstrap"] = disc_boot
    print(f"  AUC = {disc_boot['auc_mean']:.4f} ± {disc_boot['auc_std']:.4f}")

    # 2. Shower response W1
    print("Computing shower response W1...")
    sr_real = compute_shower_response(real)
    sr_gen = compute_shower_response(gen)
    w1_sr = w1_with_errors(sr_real, sr_gen, n_bootstrap=args.n_bootstrap)
    results["shower_response_w1"] = w1_sr
    print(f"  W1 = {w1_sr['w1_mean']:.4f} ± {w1_sr['w1_std']:.4f}")

    # 3. Center of energy W1
    print("Computing center of energy W1...")
    coe_real = compute_center_of_energy(real, real.shape[1])
    coe_gen = compute_center_of_energy(gen, gen.shape[1])
    w1_coe = w1_with_errors(coe_real, coe_gen, n_bootstrap=args.n_bootstrap)
    results["center_of_energy_w1"] = w1_coe
    print(f"  W1 = {w1_coe['w1_mean']:.4f} ± {w1_coe['w1_std']:.4f}")

    # 4. Layer energy W1
    print("Computing layer energy W1...")
    layers_real = compute_layer_energies(real, DATASET1_LAYER_BOUNDARIES)
    layers_gen = compute_layer_energies(gen, DATASET1_LAYER_BOUNDARIES)
    for layer_name in layers_real:
        w1_l = w1_with_errors(
            layers_real[layer_name], layers_gen[layer_name],
            n_bootstrap=args.n_bootstrap,
        )
        results[f"{layer_name}_w1"] = w1_l
        print(f"  {layer_name}: W1 = {w1_l['w1_mean']:.4f} ± {w1_l['w1_std']:.4f}")

    # Save
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
