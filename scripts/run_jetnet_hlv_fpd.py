"""Compute JetNet-paper physics-distance metrics (FPD, KPD, optionally FPND)
on flowBDT-generated JetNet samples vs real top-jet samples.

Accepts either:
  * HLV (N, 12) feature arrays  -- FPD, KPD only
  * 30x3 particle-level (N, 30, 3) -- FPD, KPD on derived jet features +
    FPND on the raw point cloud (requires torch_geometric)

Outputs a results.json + a compact LaTeX summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True, help="(N, 12) HLV npy OR (N, 30, 3) particle npy OR raw JetNet hdf5")
    p.add_argument("--gen", required=True, help="Same shape as --real (gen log-pt for 30x3 should be exp'd first)")
    p.add_argument("--gen-log-pt", action="store_true",
                   help="If --gen is (N,30,3) with log(pt_rel) in column 2, exp() it before evaluating")
    p.add_argument("--mode", choices=["hlv", "particle"], required=True)
    p.add_argument("--jet-type", default="t", help="JetNet jet type for FPND ('g','q','t','w','z')")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-events", type=int, default=50000,
                   help="Subsample real and gen to this many events (FPD recommends ~50k)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-fpnd", action="store_true", help="Skip FPND even if torch_geometric is present")
    return p.parse_args()


def load_particle(path):
    if path.endswith((".hdf5", ".h5")):
        import h5py
        with h5py.File(path, "r") as f:
            pf = np.asarray(f["particle_features"][:], dtype=np.float64)
        if pf.shape[-1] == 4:
            pf = pf[..., :3]
        return pf.astype(np.float64)
    arr = np.load(path).astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == 90:
        arr = arr.reshape(-1, 30, 3)
    if arr.ndim != 3 or arr.shape[1] != 30 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected particle shape {arr.shape}")
    return arr


def derived_jet_features(particles):
    """Compute jet-level summary features from (N, 30, 3) particles.

    Following the convention of the JetNet evaluation, we produce a small
    feature set:
      [m/pT, pT_sum, eta_rms, phi_rms, mult, energy_correlation_2]
    These serve as a 6-D physics-feature vector for FPD/KPD when EFPs
    aren't available.
    """
    eta = particles[..., 0]
    phi = particles[..., 1]
    pt = particles[..., 2]
    mask = pt > 1e-6
    pt = np.where(mask, pt, 0.0)
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta); e = pt * np.cosh(eta)
    pxs, pys, pzs, es = px.sum(-1), py.sum(-1), pz.sum(-1), e.sum(-1)
    m2 = es**2 - pxs**2 - pys**2 - pzs**2
    m_rel = np.sqrt(np.clip(m2, 0, None))
    pt_rel = np.sqrt(pxs**2 + pys**2)
    mult = mask.sum(-1).astype(np.float64)
    # weighted moments
    w = np.where(mask, pt, 0.0)
    wsum = w.sum(-1, keepdims=True).clip(1e-8)
    eta_mean = (w * eta).sum(-1, keepdims=True) / wsum
    phi_mean = (w * phi).sum(-1, keepdims=True) / wsum
    eta_rms = np.sqrt(((w * (eta - eta_mean) ** 2).sum(-1) / wsum.squeeze(-1)).clip(0))
    phi_rms = np.sqrt(((w * (phi - phi_mean) ** 2).sum(-1) / wsum.squeeze(-1)).clip(0))
    # crude ecf2
    ecf2 = ((pt[..., None] * pt[..., None, :]).sum((-2, -1)) - (pt ** 2).sum(-1))
    return np.stack([m_rel, pt_rel, eta_rms, phi_rms, mult, ecf2], axis=-1)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Load
    print(f"[load] real: {args.real}")
    print(f"[load] gen:  {args.gen}  (gen_log_pt={args.gen_log_pt})")
    if args.mode == "hlv":
        real = np.load(args.real).astype(np.float64)
        gen = np.load(args.gen).astype(np.float64)
        if real.shape[1] != gen.shape[1]:
            raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")
        feat_real = real
        feat_gen = gen
        n_features = real.shape[1]
        particle_real = particle_gen = None
    else:
        particle_real = load_particle(args.real)
        particle_gen = load_particle(args.gen)
        if args.gen_log_pt:
            particle_gen[..., 2] = np.clip(np.exp(particle_gen[..., 2]), 0, None)
        feat_real = derived_jet_features(particle_real)
        feat_gen = derived_jet_features(particle_gen)
        n_features = feat_real.shape[1]
        print(f"  particles real {particle_real.shape}  gen {particle_gen.shape}")
    print(f"  features real {feat_real.shape}  gen {feat_gen.shape}  (d={n_features})")

    # Subsample
    n = min(len(feat_real), len(feat_gen), args.n_events)
    ir = rng.permutation(len(feat_real))[:n]
    ig = rng.permutation(len(feat_gen))[:n]
    feat_real_s, feat_gen_s = feat_real[ir], feat_gen[ig]
    print(f"  subsampled to n={n}")

    from jetnet.evaluation import fpd, kpd

    results = {"config": vars(args), "n_used": n, "n_features": int(n_features)}

    # FPD
    print("\n[fpd] computing...")
    fpd_val, fpd_err = fpd(feat_real_s.astype(np.float32), feat_gen_s.astype(np.float32),
                            min_samples=min(20000, n // 2), max_samples=n, num_batches=20, num_points=10,
                            normalise=True, seed=args.seed)
    print(f"  FPD = {fpd_val:.4f} +/- {fpd_err:.4f}")
    results["fpd"] = {"value": float(fpd_val), "error": float(fpd_err)}

    # Truth-baseline FPD (split real)
    half = n // 2
    perm = rng.permutation(n)
    a, b = feat_real_s[perm[:half]], feat_real_s[perm[half:half*2]]
    fpd_t, fpd_t_err = fpd(a.astype(np.float32), b.astype(np.float32),
                            min_samples=min(20000, half), max_samples=half, num_batches=20, num_points=10,
                            normalise=True, seed=args.seed + 1)
    print(f"  FPD truth = {fpd_t:.4f} +/- {fpd_t_err:.4f}")
    results["fpd_truth"] = {"value": float(fpd_t), "error": float(fpd_t_err)}

    # KPD
    print("\n[kpd] computing...")
    kpd_val, kpd_err = kpd(feat_real_s.astype(np.float32), feat_gen_s.astype(np.float32),
                            num_batches=10, batch_size=5000, normalise=True, seed=args.seed)
    print(f"  KPD = {kpd_val:.6f} +/- {kpd_err:.6f}")
    results["kpd"] = {"value": float(kpd_val), "error": float(kpd_err)}
    kpd_t, kpd_t_err = kpd(a.astype(np.float32), b.astype(np.float32),
                            num_batches=10, batch_size=5000, normalise=True, seed=args.seed + 1)
    print(f"  KPD truth = {kpd_t:.6f} +/- {kpd_t_err:.6f}")
    results["kpd_truth"] = {"value": float(kpd_t), "error": float(kpd_t_err)}

    # FPND (only if particle mode and torch_geometric available)
    fpnd_val = None
    if args.mode == "particle" and not args.skip_fpnd:
        try:
            import torch_geometric  # noqa: F401
            from jetnet.evaluation import fpnd
            n_fpnd = min(50000, len(particle_gen))
            ig2 = rng.permutation(len(particle_gen))[:n_fpnd]
            print(f"\n[fpnd] computing on {n_fpnd} jets...")
            fpnd_val = fpnd(particle_gen[ig2].astype(np.float32), jet_type=args.jet_type, use_tqdm=False)
            print(f"  FPND = {fpnd_val:.4f}")
            results["fpnd"] = float(fpnd_val)
        except ImportError as e:
            print(f"\n[fpnd] skipped (missing dependency): {e}")
            results["fpnd"] = None

    with open(os.path.join(args.out_dir, "fpd_kpd_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # LaTeX
    lines = ["\\begin{tabular}{lcc}",
             "\\textbf{Metric} & \\textbf{flowBDT} & \\textbf{Truth baseline} \\\\",
             "\\hline",
             f"FPD & ${fpd_val:.3f} \\pm {fpd_err:.3f}$ & ${fpd_t:.3f} \\pm {fpd_t_err:.3f}$ \\\\",
             f"KPD ($\\times 10^3$) & ${kpd_val*1e3:.3f} \\pm {kpd_err*1e3:.3f}$ & ${kpd_t*1e3:.3f} \\pm {kpd_t_err*1e3:.3f}$ \\\\"]
    if fpnd_val is not None:
        lines.append(f"FPND & ${fpnd_val:.3f}$ & -- \\\\")
    lines.append("\\end{tabular}")
    with open(os.path.join(args.out_dir, "fpd_kpd_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[save] {args.out_dir}/fpd_kpd_results.json")
    print(f"[save] {args.out_dir}/fpd_kpd_table.tex")


if __name__ == "__main__":
    main()
