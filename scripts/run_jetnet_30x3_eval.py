"""JetNet 30x3 particle-level evaluation with bootstrap W1 + truth baseline.

Implements the JetNet-paper benchmark metrics asked for by the referee:

  * W1-M  : Wasserstein-1 on the jet mass (m_jet / pt_jet)
  * W1-Pt : Wasserstein-1 on the jet relative pT (sanity check, should be ~1)
  * W1-P  : Wasserstein-1 on each per-particle feature (eta_rel, phi_rel, pt_rel),
            flattened across all particles in all jets and masked

All metrics are reported as ``mean +/- std`` from a bootstrap (n=100) and
compared against a finite-sample truth baseline obtained by splitting the
real data into two halves.

Inputs
------
--real-hdf5   Raw JetNet hdf5 with particle_features of shape (N, 30, 4)
              where the 4 columns are [eta_rel, phi_rel, pt_rel, mask].
--gen-npy     Generated samples npy of shape either (N, 30, 3) or (N, 90)
              (no mask channel; we drop the mask from real to match).
              The 3 features must be in the order [eta_rel, phi_rel, pt_rel].

If --gen-npy is omitted, the script still reports the truth-baseline floor.

Outputs
-------
{out_dir}/results.json    -- all metrics as a structured JSON
{out_dir}/table.tex       -- LaTeX table matching the prd.tex style

Example
-------
    python -m BUFF.scripts.run_jetnet_30x3_eval \
        --real-hdf5 data/jetnet_raw/t.hdf5 \
        --gen-npy   results/jetnet_30x3/generated_samples.npy \
        --out-dir   rebuttal/results/jetnet_30x3/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from BUFF.evaluation.wasserstein import w1_with_errors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-hdf5", required=True, help="Raw JetNet hdf5 with particle_features (N,30,4)")
    p.add_argument("--gen-npy", default=None, help="Generated samples (N,30,3) or (N,90). Optional.")
    p.add_argument("--gen-log-pt", action="store_true",
                   help="If set, the third feature in --gen-npy is log(pt_rel); we apply exp() to recover pt_rel "
                        "(matches the build_jetnet_particle_kin.py default).")
    p.add_argument("--renormalize-pt", action="store_true",
                   help="Post-hoc: rescale each generated jet's pt_rel values so they sum to a sample drawn from "
                        "the empirical real-data pt_sum distribution. Mass-W1 is scale-invariant so this does NOT "
                        "improve W1-M, but it does fix W1-Pt and W1-P(pT). Report alongside the unmodified result.")
    p.add_argument("--out-dir", default="rebuttal/results/jetnet_30x3/")
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--n-jets", type=int, default=0,
                   help="Truncate to this many jets per side (0 = use all matched).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-particle-subsample", type=int, default=200_000,
                   help="Random subsample size for the per-particle W1 (avoids O(N^2) blowup).")
    return p.parse_args()


def load_real(path):
    with h5py.File(path, "r") as f:
        pf = np.asarray(f["particle_features"][:], dtype=np.float64)
    if pf.shape[-1] == 4:
        mask = pf[..., 3] > 0.5
        feats = pf[..., :3]
    else:
        mask = np.ones(pf.shape[:-1], dtype=bool)
        feats = pf
    return feats, mask


def load_gen(path, n_jets, log_pt=False):
    arr = np.load(path)
    if arr.ndim == 2:
        if arr.shape[1] == 90:
            arr = arr.reshape(-1, 30, 3)
        else:
            raise ValueError(f"Unexpected gen shape {arr.shape}; expected (N,30,3) or (N,90)")
    elif arr.ndim == 3:
        if arr.shape[1] != 30 or arr.shape[2] != 3:
            raise ValueError(f"Unexpected gen shape {arr.shape}; expected (N,30,3)")
    else:
        raise ValueError(f"Unexpected gen ndim {arr.ndim}")
    arr = arr.astype(np.float64)
    if log_pt:
        # Third column is log(pt_rel). Recover pt_rel; clip to >=0.
        arr[..., 2] = np.clip(np.exp(arr[..., 2]), 0.0, None)
    # Mask: particles with pt_rel <= floor are treated as zero-padding.
    mask = arr[..., 2] > 1e-6
    return arr, mask


def jet_kinematics(particles, mask):
    """Compute (m_rel, pt_rel) per jet using the JetNet convention.

    particles : (N, 30, 3) array with columns [eta_rel, phi_rel, pt_rel].
    mask      : (N, 30) bool, True for real particles.

    Returns (m_rel, pt_rel) each of shape (N,). m_rel is m_jet / pt_jet.
    """
    eta = particles[..., 0]
    phi = particles[..., 1]
    pt = np.where(mask, particles[..., 2], 0.0)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = pt * np.cosh(eta)
    px_s = px.sum(axis=-1)
    py_s = py.sum(axis=-1)
    pz_s = pz.sum(axis=-1)
    e_s = e.sum(axis=-1)
    m2 = e_s ** 2 - px_s ** 2 - py_s ** 2 - pz_s ** 2
    m_rel = np.sqrt(np.clip(m2, 0.0, None))
    pt_rel = np.sqrt(px_s ** 2 + py_s ** 2)
    return m_rel, pt_rel


def flatten_masked(particles, mask, feature_idx, subsample, rng):
    flat = particles[..., feature_idx][mask]
    if subsample > 0 and len(flat) > subsample:
        idx = rng.choice(len(flat), size=subsample, replace=False)
        flat = flat[idx]
    return flat


def make_latex_table(metrics):
    rows = [
        (r"$m_\mathrm{jet}/p_T^\mathrm{jet}$", "W1-M", metrics["jet_mass_rel"]),
        (r"$p_T^\mathrm{jet}/p_T^\mathrm{jet}$", "W1-Pt (sanity)", metrics["jet_pt_rel"]),
        (r"per-particle $\eta^\mathrm{rel}$", "W1-P($\\eta$)", metrics["particle_eta_rel"]),
        (r"per-particle $\phi^\mathrm{rel}$", "W1-P($\\phi$)", metrics["particle_phi_rel"]),
        (r"per-particle $p_T^\mathrm{rel}$", "W1-P($p_T$)", metrics["particle_pt_rel"]),
    ]
    lines = [
        r"\begin{ruledtabular}",
        r"  \begin{tabular}{lcc}",
        r"  \textbf{Variable} & \textbf{Metric} & \textbf{W1}$\times 10^3$ (truth) \\",
        r"  \hline",
    ]
    for var, label, m in rows:
        if "w1_mean" in m:
            cell = (
                f'{m["w1_mean"] * 1e3:.3f} $\\pm$ {m["w1_std"] * 1e3:.3f} '
                f'({m["truth_mean"] * 1e3:.3f} $\\pm$ {m["truth_std"] * 1e3:.3f})'
            )
        else:
            cell = (
                f'-- ({m["truth_mean"] * 1e3:.3f} $\\pm$ {m["truth_std"] * 1e3:.3f})'
            )
        lines.append(f"  {var} & {label} & {cell} \\\\")
    lines += [r"  \end{tabular}", r"\end{ruledtabular}"]
    return "\n".join(lines)


def evaluate_1d(real, gen, n_bootstrap, seed):
    if gen is None:
        from BUFF.evaluation.bootstrap import bootstrap_truth_baseline
        from BUFF.evaluation.wasserstein import _w1_1d
        truth = bootstrap_truth_baseline(real, _w1_1d, n_bootstrap=n_bootstrap, seed=seed)
        return {"truth_mean": truth["mean"], "truth_std": truth["std"]}
    return w1_with_errors(real, gen, n_bootstrap=n_bootstrap, seed=seed)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print(f"[load] real: {args.real_hdf5}")
    real_p, real_mask = load_real(args.real_hdf5)
    print(f"  particles: {real_p.shape}  mask kept: {real_mask.mean():.4f}")

    if args.gen_npy is not None:
        print(f"[load] gen:  {args.gen_npy}  (log_pt={args.gen_log_pt})")
        gen_p, gen_mask = load_gen(args.gen_npy, args.n_jets, log_pt=args.gen_log_pt)
        print(f"  particles: {gen_p.shape}  mask kept: {gen_mask.mean():.4f}")
        if args.renormalize_pt:
            # Real pt-sum statistics (use the loaded real particles, masked)
            real_pt_sum = np.where(real_mask, real_p[..., 2], 0.0).sum(axis=-1)
            gen_pt_sum = np.where(gen_mask, gen_p[..., 2], 0.0).sum(axis=-1)
            print(f"  pre-renorm gen pt_sum: mean={gen_pt_sum.mean():.4f} std={gen_pt_sum.std():.4f}")
            print(f"  real     pt_sum: mean={real_pt_sum.mean():.4f} std={real_pt_sum.std():.4f}")
            target = rng.choice(real_pt_sum, size=len(gen_pt_sum), replace=True)
            scale = np.divide(target, gen_pt_sum, out=np.ones_like(gen_pt_sum), where=gen_pt_sum > 1e-12)
            gen_p[..., 2] = gen_p[..., 2] * scale[:, None]
            new_sum = np.where(gen_mask, gen_p[..., 2], 0.0).sum(axis=-1)
            print(f"  post-renorm gen pt_sum: mean={new_sum.mean():.4f} std={new_sum.std():.4f}")
    else:
        print("[load] gen not provided -- only truth baseline will be reported")
        gen_p, gen_mask = None, None

    if args.n_jets > 0:
        real_p = real_p[: args.n_jets]
        real_mask = real_mask[: args.n_jets]
        if gen_p is not None:
            gen_p = gen_p[: args.n_jets]
            gen_mask = gen_mask[: args.n_jets]

    # match jet count: trim the larger side
    if gen_p is not None:
        n = min(len(real_p), len(gen_p))
        real_p, real_mask = real_p[:n], real_mask[:n]
        gen_p, gen_mask = gen_p[:n], gen_mask[:n]
        print(f"[match] using {n} jets per side")

    real_m, real_pt = jet_kinematics(real_p, real_mask)
    if gen_p is not None:
        gen_m, gen_pt = jet_kinematics(gen_p, gen_mask)
    else:
        gen_m = gen_pt = None

    print("[w1] jet mass...")
    jm = evaluate_1d(real_m, gen_m, args.n_bootstrap, args.seed)
    print(f"  {jm}")
    print("[w1] jet pt (sanity, should be ~0)...")
    jpt = evaluate_1d(real_pt, gen_pt, args.n_bootstrap, args.seed + 1)
    print(f"  {jpt}")

    print("[w1] per-particle (eta_rel, phi_rel, pt_rel)...")
    eta_r = flatten_masked(real_p, real_mask, 0, args.per_particle_subsample, rng)
    phi_r = flatten_masked(real_p, real_mask, 1, args.per_particle_subsample, rng)
    pt_r = flatten_masked(real_p, real_mask, 2, args.per_particle_subsample, rng)
    if gen_p is not None:
        eta_g = flatten_masked(gen_p, gen_mask, 0, args.per_particle_subsample, rng)
        phi_g = flatten_masked(gen_p, gen_mask, 1, args.per_particle_subsample, rng)
        pt_g = flatten_masked(gen_p, gen_mask, 2, args.per_particle_subsample, rng)
    else:
        eta_g = phi_g = pt_g = None

    pp_eta = evaluate_1d(eta_r, eta_g, args.n_bootstrap, args.seed + 2)
    pp_phi = evaluate_1d(phi_r, phi_g, args.n_bootstrap, args.seed + 3)
    pp_pt = evaluate_1d(pt_r, pt_g, args.n_bootstrap, args.seed + 4)

    metrics = {
        "jet_mass_rel": jm,
        "jet_pt_rel": jpt,
        "particle_eta_rel": pp_eta,
        "particle_phi_rel": pp_phi,
        "particle_pt_rel": pp_pt,
    }
    config = {
        "real_hdf5": args.real_hdf5,
        "gen_npy": args.gen_npy,
        "n_jets_used": int(len(real_p)),
        "n_bootstrap": args.n_bootstrap,
        "per_particle_subsample": args.per_particle_subsample,
        "seed": args.seed,
    }
    out = {"config": config, "metrics": metrics}

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, "table.tex"), "w") as f:
        f.write(make_latex_table(metrics) + "\n")

    print(f"[save] {args.out_dir}/results.json")
    print(f"[save] {args.out_dir}/table.tex")
    print()
    print("Summary (W1 x 1e3, gen vs real | truth floor):")
    for k, v in metrics.items():
        if "w1_mean" in v:
            print(f"  {k:22s}  {v['w1_mean']*1e3:8.3f} +- {v['w1_std']*1e3:6.3f}  |  "
                  f"{v['truth_mean']*1e3:8.3f} +- {v['truth_std']*1e3:6.3f}")
        else:
            print(f"  {k:22s}  --                    |  "
                  f"{v['truth_mean']*1e3:8.3f} +- {v['truth_std']*1e3:6.3f}")


if __name__ == "__main__":
    main()
