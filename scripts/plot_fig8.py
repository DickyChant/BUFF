"""Render Fig 8 of the BUFF paper: 6-panel unfolding comparison.

Loads four sets of samples and overlays binned histograms in each panel:
    - Target (Sim, detector-level)  -- prior baseline
    - Gen   (real particle-level)   -- truth to unfold to
    - Flow-Diffu (flowBDT from Gaussian prior)
    - Flow-OT   (flowBDT from sim prior)

Panels: jet mass, width, mult, ln(rho_SD), z_g, tau21.

Inputs are .npy arrays of shape (N, 6) in raw (un-standardized) units.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURE_LABELS = [
    r"$m_\mathrm{jet}$ [GeV]",
    r"jet width",
    r"$N_\mathrm{const}$",
    r"$\ln \rho_\mathrm{SD}$",
    r"$z_g$",
    r"$\tau_{21}$",
]
FEATURE_BINS = [
    np.linspace(0, 50, 41),
    np.linspace(0, 0.4, 41),
    np.arange(0, 70, 2),
    np.linspace(-15, 0, 41),
    np.linspace(0, 0.5, 41),
    np.linspace(0, 1.5, 41),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen",        required=True, help="Real particle-level (N,6) npy -- the unfolding target")
    p.add_argument("--sim",        required=True, help="Real detector-level (N,6) npy -- the prior")
    p.add_argument("--flow-diffu", required=True, help="Generated from Gaussian prior (N,6) npy")
    p.add_argument("--flow-ot",    required=True, help="Generated from sim prior (N,6) npy")
    p.add_argument("--out",        required=True, help="Output PDF path (e.g., fold_comp.pdf)")
    p.add_argument("--ratio-out",  default=None,  help="Optional: separate ratio-plot PDF")
    return p.parse_args()


def safe_load(path):
    arr = np.load(path).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"{path}: expected (N, 6), got {arr.shape}")
    return arr


def plot_comparison(gen, sim, fd, fot, out_path, with_ratio=True):
    n_panels = 6
    ncols = 3
    nrows = 2

    if with_ratio:
        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(15, 9),
                                 gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.04})
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))

    for i in range(n_panels):
        row = (i // ncols) * (2 if with_ratio else 1)
        col = i % ncols
        if with_ratio:
            ax_main = axes[row, col]
            ax_ratio = axes[row + 1, col]
        else:
            ax_main = axes[row, col]
            ax_ratio = None

        bins = FEATURE_BINS[i]
        kw = dict(bins=bins, histtype="step", density=True, lw=1.6)
        h_gen, _ = np.histogram(gen[:, i], bins=bins, density=True)
        h_sim, _ = np.histogram(sim[:, i], bins=bins, density=True)
        h_fd, _ = np.histogram(fd[:, i], bins=bins, density=True)
        h_fot, _ = np.histogram(fot[:, i], bins=bins, density=True)

        ax_main.hist(gen[:, i], color="black",   linestyle="-",  label="Gen (target)", **kw)
        ax_main.hist(sim[:, i], color="grey",    linestyle=":",  label="Sim (prior)", **kw)
        ax_main.hist(fd[:, i],  color="green",   linestyle="-",  label="Flow-Diffu", **kw)
        ax_main.hist(fot[:, i], color="#1f77b4", linestyle="-",  label="Flow-OT", **kw)

        ax_main.set_ylabel("A.U.", fontsize=12)
        if i == 0:
            ax_main.legend(fontsize=9, loc="upper right", frameon=False)
        ax_main.tick_params(axis="both", labelsize=10)

        if with_ratio:
            # Ratios w.r.t. Gen target; avoid divide-by-zero
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            denom = np.where(h_gen > 1e-12, h_gen, 1.0)
            r_sim = h_sim / denom
            r_fd = h_fd / denom
            r_fot = h_fot / denom
            ax_ratio.plot(bin_centers, r_sim, color="grey",    linestyle=":")
            ax_ratio.plot(bin_centers, r_fd,  color="green",   linestyle="-")
            ax_ratio.plot(bin_centers, r_fot, color="#1f77b4", linestyle="-")
            ax_ratio.axhline(1.0, color="black", lw=0.5, alpha=0.5)
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.set_ylabel("ratio", fontsize=10)
            ax_ratio.set_xlabel(FEATURE_LABELS[i], fontsize=12)
            ax_ratio.tick_params(axis="both", labelsize=9)
            ax_main.tick_params(labelbottom=False)
        else:
            ax_main.set_xlabel(FEATURE_LABELS[i], fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main():
    args = parse_args()
    print("[load] gen   :", args.gen)
    gen = safe_load(args.gen)
    print("[load] sim   :", args.sim)
    sim = safe_load(args.sim)
    print("[load] diffu :", args.flow_diffu)
    fd = safe_load(args.flow_diffu)
    print("[load] flowot:", args.flow_ot)
    fot = safe_load(args.flow_ot)

    # Match sample counts for fair density comparison
    n = min(len(gen), len(sim), len(fd), len(fot))
    gen, sim, fd, fot = gen[:n], sim[:n], fd[:n], fot[:n]
    print(f"  using {n} events per dataset")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plot_comparison(gen, sim, fd, fot, args.out, with_ratio=True)

    if args.ratio_out:
        # Optional: also a comp-only (no ratio) version
        plot_comparison(gen, sim, fd, fot, args.ratio_out, with_ratio=False)


if __name__ == "__main__":
    main()
