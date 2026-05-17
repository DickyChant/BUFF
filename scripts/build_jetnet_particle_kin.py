"""Build a leading-N particle kinematic representation from raw JetNet-30.

For each jet, sort its 30 constituent particles by pt_rel and keep the
top N (default 4).  Output flattened (N * 3) features per jet:

    [η_rel¹, φ_rel¹, log pt_rel¹,  η_rel², ..., log pt_rel^N]

`log pt_rel` is used because pt_rel is heavy-tailed; log compresses the
dynamic range and matches the Gaussian-ish base distribution F2D2 expects.

Usage
-----
    python -m BUFF.scripts.build_jetnet_particle_kin \
        --raw /path/to/t.hdf5 \
        --out /path/to/t_p4_kin.npy \
        --n-leading 4
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Raw JetNet hdf5 with particle_features")
    p.add_argument("--out", required=True, help="Output .npy")
    p.add_argument("--n-leading", type=int, default=4)
    p.add_argument("--pt-floor", type=float, default=1e-6,
                   help="Floor on pt_rel before taking log (avoids -inf for masked particles)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.raw}")
    with h5py.File(args.raw, "r") as f:
        pf = f["particle_features"][:].astype(np.float64)
    N, P, F = pf.shape
    print(f"  particle_features: {pf.shape}  ({N} jets × {P} particles × {F} features)")

    eta = pf[..., 0]
    phi = pf[..., 1]
    pt = pf[..., 2]
    mask = pf[..., 3]

    # Zero-out masked particles' pt so they sort to the bottom.
    pt_masked = pt * (mask > 0)

    # argsort descending by pt; take leading N.
    order = np.argsort(-pt_masked, axis=1)[:, : args.n_leading]
    rows = np.arange(N)[:, None]
    eta_lead = eta[rows, order]
    phi_lead = phi[rows, order]
    pt_lead = pt_masked[rows, order]
    log_pt_lead = np.log(np.clip(pt_lead, args.pt_floor, None))

    # Stack (N, n_leading, 3): (eta, phi, log_pt)
    features = np.stack([eta_lead, phi_lead, log_pt_lead], axis=-1)
    flat = features.reshape(N, -1).astype(np.float32)

    print(f"  output shape: {flat.shape}  dtype={flat.dtype}")
    print(f"  per-column stats:")
    for k in range(flat.shape[1]):
        c = flat[:, k]
        which = ["eta", "phi", "log_pt"][k % 3]
        slot = k // 3 + 1
        print(f"    p{slot:>1} {which:<7}  [{c.min():+.4e}, {c.max():+.4e}]  mean={c.mean():+.4e}")

    bad = ~np.isfinite(flat).all(axis=1)
    if bad.any():
        print(f"  dropping {int(bad.sum())} non-finite rows ({100*bad.sum()/N:.2f}%)")
        flat = flat[~bad]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, flat)
    print(f"Wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
