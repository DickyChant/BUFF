"""Build the Omnifold (Z+Jets, Delphes) gen/sim feature pair for unfolding.

Produces six high-level observables per jet, following the convention of the
existing `preprocessing/preprocessing_omnifold.py` (which is the recipe used
to produce Figure 8 of the original BUFF paper):

    col 0: jet mass           (m)
    col 1: jet width          (beta=1 angularity)
    col 2: constituent mult.  (N)
    col 3: ln(rho_SD) = 2*log(m_SD / p_T^jet)
    col 4: z_g                (Soft Drop groomed momentum fraction)
    col 5: tau21              = tau_2 / width  (beta=1 angularity = tau_1)

Inputs are downloaded once via the `energyflow` package (cached in cache-dir).

Outputs
-------
{out_dir}/{tag}_gen_features.npy   (N, 6)  particle-level
{out_dir}/{tag}_sim_features.npy   (N, 6)  detector-level (matched index-wise)
{out_dir}/{tag}_scaler.json        dict with mean / std per feature for
                                    inversion; both gen and sim are stored
                                    in *raw* units (no standardization),
                                    leaving rescaling to train_and_sample.

Selection follows the paper: drop jets outside reasonable physical bounds on
each observable to remove outliers from poorly-clustered events.
"""

from __future__ import annotations

import argparse
import json
import os

import energyflow as ef
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="Pythia26", choices=["Pythia21", "Pythia25", "Pythia26", "Herwig"])
    p.add_argument("--n-events", type=int, default=500_000,
                   help="Number of events to load (gen+sim matched).")
    p.add_argument("--cache-dir", default="/pscratch/sd/s/sqian/buff_f2d2/data/omnifold/cache")
    p.add_argument("--out-dir", default="/pscratch/sd/s/sqian/buff_f2d2/data/omnifold")
    p.add_argument("--tag", default="pythia26")
    return p.parse_args()


def build_features(d, side):
    """side in {'gen', 'sim'}.  Returns (N, 6) float32."""
    jets = d[side + "_jets"]
    mass = jets[:, 3]
    pt = jets[:, 0]
    widths = d[side + "_widths"]
    mults = d[side + "_mults"]
    sdms = d[side + "_sdms"]
    zgs = d[side + "_zgs"]
    tau2 = d[side + "_tau2s"]

    eps = 1e-50
    # ln rho_SD = 2 * log(m_SD / pt_jet); use masked array to safely handle m_SD == 0
    rho = np.where(sdms > 0, sdms, eps) / np.where(pt > 0, pt, eps)
    ln_rho = 2.0 * np.log(np.where(rho > 0, rho, eps))
    # tau21 = tau2 / width  (width is the beta=1 angularity, equal to tau_1)
    tau21 = tau2 / (eps + widths)

    out = np.stack([mass, widths, mults, ln_rho, zgs, tau21], axis=-1).astype(np.float32)
    return out


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] sample={args.sample}  n_events={args.n_events}  cache={args.cache_dir}")
    d = ef.zjets_delphes.load(
        args.sample,
        num_data=args.n_events,
        cache_dir=args.cache_dir,
        exclude_keys=["particles"],
    )
    print("  loaded keys:", sorted([k for k in d.keys() if not k.startswith("_")]))

    gen = build_features(d, "gen")
    sim = build_features(d, "sim")
    print(f"  gen shape: {gen.shape}   sim shape: {sim.shape}")

    # Loose physics selection cut from preprocessing_omnifold.py
    sel = (
        (gen[:, 0] < 100) & (sim[:, 0] < 100)     # jet mass under 100 GeV (very loose)
        & (gen[:, 1] < 0.6) & (sim[:, 1] < 0.6)   # widths in [0, 0.6]
        & (gen[:, 2] < 100) & (sim[:, 2] < 100)   # multiplicity sane
        & (gen[:, 3] > -20) & (gen[:, 3] < 0)     # ln(rho) in physical range
        & (sim[:, 3] > -20) & (sim[:, 3] < 0)
        & (gen[:, 4] >= 0) & (gen[:, 4] <= 0.5)   # z_g in [0, 0.5]
        & (sim[:, 4] >= 0) & (sim[:, 4] <= 0.5)
        & (gen[:, 5] >= 0) & (gen[:, 5] <= 5)     # tau21 sane
        & (sim[:, 5] >= 0) & (sim[:, 5] <= 5)
        & np.isfinite(gen).all(axis=-1) & np.isfinite(sim).all(axis=-1)
    )
    print(f"  selection kept {sel.sum()}/{len(sel)} = {sel.mean()*100:.1f}%")
    gen = gen[sel]
    sim = sim[sel]

    # Per-feature mean/std on raw units (for later re-scaling / plotting)
    stats = {
        "feature_names": ["mass", "width", "mult", "ln_rho", "zg", "tau21"],
        "gen": {
            "mean": gen.mean(axis=0).tolist(),
            "std": gen.std(axis=0).tolist(),
            "min": gen.min(axis=0).tolist(),
            "max": gen.max(axis=0).tolist(),
        },
        "sim": {
            "mean": sim.mean(axis=0).tolist(),
            "std": sim.std(axis=0).tolist(),
            "min": sim.min(axis=0).tolist(),
            "max": sim.max(axis=0).tolist(),
        },
        "n_events": int(len(gen)),
        "sample": args.sample,
    }

    gen_path = os.path.join(args.out_dir, f"{args.tag}_gen_features.npy")
    sim_path = os.path.join(args.out_dir, f"{args.tag}_sim_features.npy")
    stats_path = os.path.join(args.out_dir, f"{args.tag}_stats.json")
    np.save(gen_path, gen)
    np.save(sim_path, sim)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[save] {gen_path}")
    print(f"[save] {sim_path}")
    print(f"[save] {stats_path}")
    print()
    print("Per-feature gen stats (raw units):")
    for i, name in enumerate(stats["feature_names"]):
        print(f"  {name:8s}  mean={stats['gen']['mean'][i]:+.4f}  std={stats['gen']['std'][i]:.4f}  "
              f"range=[{stats['gen']['min'][i]:+.4f}, {stats['gen']['max'][i]:+.4f}]")


if __name__ == "__main__":
    main()
