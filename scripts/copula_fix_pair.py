"""Post-hoc Gaussian-copula correction targeted at a specific feature pair.

Given the diagnostic finding that the (d_{23}, d_2) joint distribution drives
~all of the discriminator-AUC signal (single pair-AUC ~0.85 while all other
65 pairs sit at AUC < 0.68), this script applies a minimal targeted fix:

  1. Convert each feature i of (real, gen) to its empirical rank-uniform
     via probability integral transform (PIT)
  2. Map to standard normal via Phi^{-1}
  3. In normal space, compute the Pearson correlation of the chosen pair
     (i, j) in real and gen
  4. Linearly transform gen so that its (i, j) covariance in normal space
     matches the real (i, j) covariance, keeping each *marginal* (in
     normal space) standard normal
  5. Inverse-PIT each feature back to the real marginal CDF

This preserves all marginals exactly (per-feature AUCs unchanged) and only
adjusts the targeted joint.  Other pair-correlations involving i or j may
shift slightly; this is acceptable because (a) those other pairs are
already at AUC ~0.5-0.6 (well-modelled), (b) the joint we're fixing is the
single dominant signal.

Usage
-----
    python -m BUFF.scripts.copula_fix_pair \
        --real /path/to/t_hlv12.npy \
        --gen  /path/to/generated_samples.npy \
        --out  /path/to/corrected_samples.npy \
        --feat-i d_23 --feat-j d_2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import norm, rankdata


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


FEATURE_NAMES = ["d_12", "d_23", "mass", "pt", "tau_1", "tau_2", "tau_3",
                 "tau_21", "tau_32", "ecf_2", "ecf_3", "d_2"]
NAME_TO_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feat-i", default="d_23", choices=FEATURE_NAMES)
    p.add_argument("--feat-j", default="d_2",  choices=FEATURE_NAMES)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def to_uniform(x):
    """Convert to ranks normalized to (0, 1), excluding the open endpoints."""
    n = len(x)
    r = rankdata(x, method="average")
    return (r - 0.5) / n


def correct_pair_gaussian_copula(real, gen, i, j):
    """Match the (i, j) gauss-copula correlation of gen to that of real.

    Returns the corrected gen array (same shape, marginals preserved).
    """
    gen_out = gen.copy().astype(np.float64)
    real = real.astype(np.float64)

    # Real reference: rho_real in normal space
    real_u_i = to_uniform(real[:, i])
    real_u_j = to_uniform(real[:, j])
    real_z_i = norm.ppf(real_u_i)
    real_z_j = norm.ppf(real_u_j)
    rho_real = float(np.corrcoef(real_z_i, real_z_j)[0, 1])

    # Gen: same PIT + ppf
    gen_u_i = to_uniform(gen[:, i])
    gen_u_j = to_uniform(gen[:, j])
    gen_z_i = norm.ppf(gen_u_i).astype(np.float64)
    gen_z_j = norm.ppf(gen_u_j).astype(np.float64)
    rho_gen = float(np.corrcoef(gen_z_i, gen_z_j)[0, 1])

    print(f"  rho_real (normal space, features {i},{j}) = {rho_real:+.4f}")
    print(f"  rho_gen  (normal space, features {i},{j}) = {rho_gen:+.4f}")

    # Construct linear map A s.t. (z_i, z_j_new) has cov = [[1, rho_real],[rho_real, 1]]
    # while keeping z_i standard normal AND z_j_new standard normal.
    # Whiten gen first:
    #   W = L_gen^{-1} where L_gen is Cholesky of cov(z_i, z_j) (= [[1, rho_gen],[rho_gen,1]])
    # Then apply L_real:
    #   new = L_real @ W @ old
    # Construct a Z = stack of (z_i, z_j); apply transform.
    Z = np.stack([gen_z_i, gen_z_j], axis=0)  # shape (2, N)

    def chol2(rho):
        # Cholesky of [[1, rho],[rho, 1]] => [[1, 0],[rho, sqrt(1-rho^2)]]
        return np.array([[1.0, 0.0], [rho, np.sqrt(max(0.0, 1.0 - rho * rho))]])

    L_gen = chol2(rho_gen)
    L_real = chol2(rho_real)
    A = L_real @ np.linalg.inv(L_gen)
    Z_new = A @ Z
    # Verify
    rho_new = float(np.corrcoef(Z_new[0], Z_new[1])[0, 1])
    print(f"  rho_gen (after transform)              = {rho_new:+.4f}")

    # Inverse PIT back through the *real* marginal CDF -- this preserves
    # each gen marginal as it was (assuming the orig gen marginal already
    # matched real well enough).  We use the corrected ranks against the
    # SAME real distribution by quantile-matching.
    u_i_new = norm.cdf(Z_new[0])
    u_j_new = norm.cdf(Z_new[1])

    # Map u back to the gen feature's empirical distribution -- this
    # preserves the gen marginal exactly.  Use sorted gen values as the
    # empirical inverse CDF.
    def empirical_quantile(values, u):
        sorted_v = np.sort(values)
        idx = np.clip((u * len(sorted_v)).astype(int), 0, len(sorted_v) - 1)
        return sorted_v[idx]

    gen_out[:, i] = empirical_quantile(gen[:, i], u_i_new)
    gen_out[:, j] = empirical_quantile(gen[:, j], u_j_new)
    return gen_out, dict(rho_real=rho_real, rho_gen_before=rho_gen, rho_gen_after=rho_new)


def main():
    args = parse_args()
    i = NAME_TO_IDX[args.feat_i]
    j = NAME_TO_IDX[args.feat_j]
    np.random.seed(args.seed)

    print(f"[load] real: {args.real}")
    real = np.load(args.real).astype(np.float32)
    print(f"[load] gen:  {args.gen}")
    gen = np.load(args.gen).astype(np.float32)
    print(f"[fix] feature pair ({args.feat_i} idx {i}, {args.feat_j} idx {j})")

    gen_out, info = correct_pair_gaussian_copula(real, gen, i, j)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, gen_out.astype(np.float32))
    print(f"[save] {args.out}")
    print(f"\nrho_real    = {info['rho_real']:+.4f}")
    print(f"rho_gen_old = {info['rho_gen_before']:+.4f}")
    print(f"rho_gen_new = {info['rho_gen_after']:+.4f}")


if __name__ == "__main__":
    main()
