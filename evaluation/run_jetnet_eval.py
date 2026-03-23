"""Full JetNet high-level evaluation with bootstrap uncertainties.

Usage
-----
uv run python -m BUFF.evaluation.run_jetnet_eval \
    --real real_samples.npy --gen generated_samples.npy \
    --features d12,d2,mass,pt,tau1,tau2,tau3,tau21,tau32,ecf2,ecf3,d2_obs \
    --output-dir results/jetnet_eval/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from BUFF.evaluation.wasserstein import w1_with_errors
from BUFF.evaluation.metrics import sep_power_with_errors
from BUFF.evaluation.consistency import tau21_consistency_check, tau32_consistency_check


FEATURE_NAMES_12 = [
    "d12", "d2", "mass", "pt",
    "tau1", "tau2", "tau3",
    "tau21", "tau32",
    "ecf2", "ecf3", "d2_obs",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="Path to real samples .npy")
    p.add_argument("--gen", required=True, help="Path to generated samples .npy")
    p.add_argument(
        "--features",
        type=str,
        default=",".join(FEATURE_NAMES_12),
        help="Comma-separated feature names (must match column order)",
    )
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--output-dir", type=str, default="results/jetnet_eval/")
    return p.parse_args()


def make_latex_table(results, feature_names):
    """Generate a LaTeX table string from evaluation results."""
    lines = []
    lines.append(r"\begin{ruledtabular}")
    lines.append(r"  \begin{tabular}{ccc}")
    lines.append(
        r"  \textbf{Metrics} & \textbf{Sep power}$\times$100 & \textbf{W1}$\times$10 \\"
    )
    lines.append(r"  \hline")
    for name in feature_names:
        r = results[name]
        sep_str = f'{r["sep_mean"] * 100:.3f} $\\pm$ {r["sep_std"] * 100:.3f}'
        sep_truth = f'({r["sep_truth_mean"] * 100:.3f} $\\pm$ {r["sep_truth_std"] * 100:.3f})'
        w1_str = f'{r["w1_mean"] * 10:.4f} $\\pm$ {r["w1_std"] * 10:.4f}'
        w1_truth = f'({r["w1_truth_mean"] * 10:.4f} $\\pm$ {r["w1_truth_std"] * 10:.4f})'
        lines.append(
            f"  \\textbf{{${name}$}} & {sep_str} {sep_truth} & {w1_str} {w1_truth} \\\\"
        )
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{ruledtabular}")
    return "\n".join(lines)


def plot_consistency(check_result, name, output_path):
    """Plot generated vs derived ratio for consistency check."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bins = np.linspace(0, 2, 50)
    ax.hist(check_result["direct_" + name], bins=bins, histtype="step",
            label=f"Generated {name}", density=True, linewidth=1.5)
    ax.hist(check_result["derived_" + name], bins=bins, histtype="step",
            label=f"Derived {name}", density=True, linewidth=1.5, linestyle="--")
    ax.set_xlabel(name)
    ax.set_ylabel("Normalised")
    ax.legend()
    ax.set_title(f"Consistency: W1 = {check_result['w1']:.4f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real = np.load(args.real)
    gen = np.load(args.gen)
    feature_names = args.features.split(",")

    assert real.shape[1] == len(feature_names), (
        f"Real data has {real.shape[1]} columns but {len(feature_names)} feature names given"
    )
    assert gen.shape[1] == len(feature_names), (
        f"Gen data has {gen.shape[1]} columns but {len(feature_names)} feature names given"
    )

    results = {}
    print(f"Evaluating {len(feature_names)} features with {args.n_bootstrap} bootstrap iterations...")

    for i, name in enumerate(feature_names):
        print(f"  [{i + 1}/{len(feature_names)}] {name}")
        w1 = w1_with_errors(real[:, i], gen[:, i], n_bootstrap=args.n_bootstrap)
        sep = sep_power_with_errors(real[:, i], gen[:, i], n_bootstrap=args.n_bootstrap)
        results[name] = {
            "w1_mean": w1["w1_mean"], "w1_std": w1["w1_std"],
            "w1_truth_mean": w1["truth_mean"], "w1_truth_std": w1["truth_std"],
            "sep_mean": sep["sep_mean"], "sep_std": sep["sep_std"],
            "sep_truth_mean": sep["truth_mean"], "sep_truth_std": sep["truth_std"],
        }

    # Consistency checks (if tau columns present)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    consistency = {}
    if all(n in name_to_idx for n in ("tau1", "tau2", "tau21")):
        c21 = tau21_consistency_check(
            gen[:, name_to_idx["tau1"]],
            gen[:, name_to_idx["tau2"]],
            gen[:, name_to_idx["tau21"]],
        )
        consistency["tau21"] = {"w1": c21["w1"]}
        plot_consistency(c21, "tau21", os.path.join(args.output_dir, "consistency_tau21.pdf"))
        print(f"  tau21 consistency W1 = {c21['w1']:.4f}")

    if all(n in name_to_idx for n in ("tau2", "tau3", "tau32")):
        c32 = tau32_consistency_check(
            gen[:, name_to_idx["tau2"]],
            gen[:, name_to_idx["tau3"]],
            gen[:, name_to_idx["tau32"]],
        )
        consistency["tau32"] = {"w1": c32["w1"]}
        plot_consistency(c32, "tau32", os.path.join(args.output_dir, "consistency_tau32.pdf"))
        print(f"  tau32 consistency W1 = {c32['w1']:.4f}")

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"metrics": results, "consistency": consistency}, f, indent=2)

    latex = make_latex_table(results, feature_names)
    with open(os.path.join(args.output_dir, "table.tex"), "w") as f:
        f.write(latex)

    print(f"\nResults saved to {args.output_dir}")
    print("\nLaTeX table:\n")
    print(latex)


if __name__ == "__main__":
    main()
