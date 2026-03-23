"""Preprocess JetNet data → 12 high-level features using fastjet.

Fallback for preprocess_jetnet.py when pyjet is unavailable.
Computes: d12, d2, mass, pt, tau1, tau2, tau3, tau21, tau32, ecf2, ecf3, d2_obs

Output column order matches FEATURE_NAMES_12 in run_jetnet_eval.py:
    d12, d2, mass, pt, tau1, tau2, tau3, tau21, tau32, ecf2, ecf3, d2_obs
"""

import argparse

import awkward as ak
import fastjet
import h5py
import numpy as np
from tqdm import tqdm


# ── Substructure helpers ──────────────────────────────────────────────


def _delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = np.abs(phi1 - phi2)
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    return np.sqrt(deta**2 + dphi**2)


def _px_py_pz_E(pts, etas, phis, masses=None):
    """Convert (pt, eta, phi, mass) to (px, py, pz, E)."""
    px = pts * np.cos(phis)
    py = pts * np.sin(phis)
    pz = pts * np.sinh(etas)
    if masses is not None:
        E = np.sqrt(px**2 + py**2 + pz**2 + masses**2)
    else:
        E = np.sqrt(px**2 + py**2 + pz**2)
    return px, py, pz, E


def _jet_eta_phi_pt(jet_4vec):
    """Extract eta, phi, pt from a 4-vector dict with px, py, pz, E."""
    px = np.asarray(jet_4vec["px"])
    py = np.asarray(jet_4vec["py"])
    pz = np.asarray(jet_4vec["pz"])
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.arctanh(np.clip(pz / np.where(p > 0, p, 1), -1 + 1e-10, 1 - 1e-10))
    phi = np.arctan2(py, px)
    return eta, phi, pt


def _cluster_jet(pts, etas, phis, R=0.8):
    """Cluster particles with kt algorithm, return ClusterSequence."""
    px, py, pz, E = _px_py_pz_E(pts, etas, phis)
    particles = ak.zip({"px": px, "py": py, "pz": pz, "E": E})
    particles = ak.Array([particles])
    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R)
    return fastjet.ClusterSequence(particles, jetdef)


def _exclusive_axes(cluster, n_axes):
    """Get N exclusive jet axes as list of (eta, phi) tuples."""
    try:
        jets = cluster.exclusive_jets(n_jets=n_axes)
        px = np.asarray(jets.px[0])
        py = np.asarray(jets.py[0])
        pz = np.asarray(jets.pz[0])
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        eta = np.arctanh(np.clip(pz / np.where(p > 0, p, 1), -1 + 1e-10, 1 - 1e-10))
        phi = np.arctan2(py, px)
        return list(zip(eta, phi))
    except Exception:
        return None


def _nsubjettiness(cnst_pts, cnst_etas, cnst_phis, axes, R=0.8):
    """N-subjettiness: tau_N = (1/d0) sum_i pT_i * min_j dR(i, axis_j)."""
    d0 = np.sum(cnst_pts) * R
    if d0 == 0:
        return 0.0
    min_drs = np.full(len(cnst_pts), np.inf)
    for ax_eta, ax_phi in axes:
        dr = _delta_r(cnst_etas, cnst_phis, ax_eta, ax_phi)
        min_drs = np.minimum(min_drs, dr)
    return float(np.sum(cnst_pts * min_drs) / d0)


def _ecf2_vectorized(cnst_pts, cnst_etas, cnst_phis, pt_sum):
    """Degree-2 energy correlation function (vectorized)."""
    n = len(cnst_pts)
    if n < 2 or pt_sum == 0:
        return 0.0
    # Create upper-triangle index pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    dr = _delta_r(cnst_etas[i_idx], cnst_phis[i_idx],
                  cnst_etas[j_idx], cnst_phis[j_idx])
    total = np.sum(cnst_pts[i_idx] * cnst_pts[j_idx] * dr)
    return float(total / (pt_sum * pt_sum))


def _ecf3_vectorized(cnst_pts, cnst_etas, cnst_phis, pt_sum):
    """Degree-3 energy correlation function (vectorized for speed)."""
    n = len(cnst_pts)
    if n < 3 or pt_sum == 0:
        return 0.0

    # For up to ~30 particles, the vectorized approach is manageable
    # Generate all (i,j,k) triplets with i<j<k
    indices = np.arange(n)
    idx = np.array(np.meshgrid(indices, indices, indices)).T.reshape(-1, 3)
    idx = idx[(idx[:, 0] < idx[:, 1]) & (idx[:, 1] < idx[:, 2])]

    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    dr_ij = _delta_r(cnst_etas[i], cnst_phis[i], cnst_etas[j], cnst_phis[j])
    dr_jk = _delta_r(cnst_etas[j], cnst_phis[j], cnst_etas[k], cnst_phis[k])
    dr_ki = _delta_r(cnst_etas[k], cnst_phis[k], cnst_etas[i], cnst_phis[i])
    total = np.sum(cnst_pts[i] * cnst_pts[j] * cnst_pts[k] * dr_ij * dr_jk * dr_ki)
    return float(total / (pt_sum**3))


# ── Per-jet computation ──────────────────────────────────────────────


def compute_jet_features(particles, R=0.8):
    """Compute 12 high-level features for a single jet.

    particles: (n_particles, >=3) with columns [eta_rel, phi_rel, pt, ...]
    Returns dict with 12 features.
    """
    # Filter zero-padded particles (check first 3 cols only)
    mask = np.any(particles[:, :3] != 0, axis=-1)
    eta = particles[mask, 0].astype(np.float64)
    phi = particles[mask, 1].astype(np.float64)
    pt = particles[mask, 2].astype(np.float64)
    n_valid = int(mask.sum())

    zeros = {k: 0.0 for k in [
        "d12", "d2", "mass", "pt", "tau1", "tau2", "tau3",
        "tau21", "tau32", "ecf2", "ecf3", "d2_obs"
    ]}
    if n_valid < 2:
        return zeros

    # Jet kinematics
    jet_px = np.sum(pt * np.cos(phi))
    jet_py = np.sum(pt * np.sin(phi))
    jet_pz = np.sum(pt * np.sinh(eta))
    jet_E = np.sum(pt * np.cosh(eta))
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_m = np.sqrt(max(jet_E**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0))

    pt_sum = np.sum(pt)

    # Cluster once with kt algorithm
    cluster = _cluster_jet(pt, eta, phi, R)

    # N-subjettiness via exclusive kt axes
    axes1 = _exclusive_axes(cluster, 1)
    axes2 = _exclusive_axes(cluster, 2)
    axes3 = _exclusive_axes(cluster, 3)

    tau1 = _nsubjettiness(pt, eta, phi, axes1, R) if axes1 else 0.0
    tau2 = _nsubjettiness(pt, eta, phi, axes2, R) if axes2 else 0.0
    tau3 = _nsubjettiness(pt, eta, phi, axes3, R) if axes3 else 0.0
    tau21 = tau2 / tau1 if tau1 > 0 else 0.0
    tau32 = tau3 / tau2 if tau2 > 0 else 0.0

    # Splitting scales via exclusive_dmerge
    try:
        dm1 = cluster.exclusive_dmerge(1)
        d12_val = np.sqrt(max(float(np.asarray(dm1).flat[0]), 0))
    except Exception:
        d12_val = 0.0
    try:
        dm2 = cluster.exclusive_dmerge(2)
        d23_val = np.sqrt(max(float(np.asarray(dm2).flat[0]), 0))
    except Exception:
        d23_val = 0.0

    # Energy correlation functions
    ecf2_val = _ecf2_vectorized(pt, eta, phi, pt_sum)
    ecf3_val = _ecf3_vectorized(pt, eta, phi, pt_sum)

    # D2 observable = ecf3 * ecf1 / ecf2^2  (ecf1 normalised = pt_sum)
    d2_obs = (ecf3_val * pt_sum) / (ecf2_val**2) if ecf2_val > 0 else 0.0

    return {
        "d12": d12_val, "d2": d23_val, "mass": jet_m, "pt": jet_pt,
        "tau1": tau1, "tau2": tau2, "tau3": tau3,
        "tau21": tau21, "tau32": tau32,
        "ecf2": ecf2_val, "ecf3": ecf3_val, "d2_obs": d2_obs,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Preprocess JetNet → 12 HLVs (fastjet)")
    parser.add_argument("--input", "-f", required=True, help="Path to JetNet HDF5 file")
    parser.add_argument("--output", "-o", default=None, help="Output .npy path")
    parser.add_argument("--max-jets", "-n", type=int, default=0, help="Max jets (0=all)")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        data = f["particle_features"][()]
        data = data.astype(np.float32)
    print(f"Loaded {data.shape[0]} jets, {data.shape[1]} particles, {data.shape[2]} features")

    if args.max_jets > 0:
        data = data[: args.max_jets]
        print(f"Using first {len(data)} jets")

    feature_order = [
        "d12", "d2", "mass", "pt", "tau1", "tau2", "tau3",
        "tau21", "tau32", "ecf2", "ecf3", "d2_obs",
    ]

    results = np.zeros((len(data), len(feature_order)))
    for idx in tqdm(range(len(data)), desc="Processing jets"):
        feats = compute_jet_features(data[idx])
        for fi, fname in enumerate(feature_order):
            results[idx, fi] = feats[fname]

    output_path = args.output or args.input.replace(".hdf5", "_hlv12.npy").replace(".h5", "_hlv12.npy")
    np.save(output_path, results)
    print(f"Saved {results.shape} to {output_path}")

    # Print summary statistics
    for fi, fname in enumerate(feature_order):
        col = results[:, fi]
        print(f"  {fname:>8s}: mean={col.mean():.6f}  std={col.std():.6f}  "
              f"min={col.min():.6f}  max={col.max():.6f}")


if __name__ == "__main__":
    main()
