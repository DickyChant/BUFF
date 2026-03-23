"""Preprocess CaloChallenge Dataset 1 → high-level features for flowBDT training.

Instead of training on all 368 voxels (too expensive for BDT),
extract high-level features: total energy, layer energies, shower shape.

Also saves raw showers for discriminator evaluation.
"""

import argparse
import os

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# CaloChallenge dataset 1 photon layer boundaries (368 voxels total)
# From binning_dataset_1_photons.xml:
# Layer 0: voxels 0-2 (3 voxels)
# Layer 1: voxels 3-14 (12 voxels)
# Layer 2: voxels 15-287 (273 voxels = 3 radial x 91 alpha)
# Layer 3: voxels 288-331 (44 voxels)
# Layer 4: voxels 332-367 (36 voxels)
LAYER_BOUNDARIES = [
    (0, 3),       # Layer 0
    (3, 15),      # Layer 1
    (15, 288),    # Layer 2
    (288, 332),   # Layer 3
    (332, 368),   # Layer 4
]


def extract_hlv(showers, energies):
    """Extract high-level features from raw shower data.

    Returns array of shape (N, n_features) with:
    - total deposited energy / incident energy
    - 5 layer energy fractions
    - shower depth (energy-weighted mean layer)
    - shower width (energy-weighted std of voxel index)
    """
    N = len(showers)
    total_E = showers.sum(axis=1)

    # Layer energies
    layer_Es = np.zeros((N, len(LAYER_BOUNDARIES)))
    for i, (start, end) in enumerate(LAYER_BOUNDARIES):
        layer_Es[:, i] = showers[:, start:end].sum(axis=1)

    # Layer energy fractions
    total_E_safe = np.where(total_E > 0, total_E, 1.0)
    layer_fracs = layer_Es / total_E_safe[:, None]

    # Energy response: total deposited / incident
    energies_safe = np.where(energies > 0, energies, 1.0)
    response = total_E / energies_safe

    # Shower depth: energy-weighted mean layer index
    layer_centers = np.array([0.5 * (s + e) for s, e in LAYER_BOUNDARIES])
    depth = (layer_Es * layer_centers[None, :]).sum(axis=1) / total_E_safe

    # Shower width: energy-weighted std of voxel index
    voxel_idx = np.arange(368)
    weighted_mean = (showers * voxel_idx[None, :]).sum(axis=1) / total_E_safe
    weighted_var = (showers * (voxel_idx[None, :] - weighted_mean[:, None])**2).sum(axis=1) / total_E_safe
    width = np.sqrt(np.maximum(weighted_var, 0))

    # Combine features
    features = np.column_stack([
        response,           # 1: total energy response
        layer_fracs,        # 5: layer energy fractions
        depth,              # 1: shower depth
        width,              # 1: shower width
    ])
    return features


FEATURE_NAMES = [
    "response", "layer0_frac", "layer1_frac", "layer2_frac",
    "layer3_frac", "layer4_frac", "depth", "width",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-f", required=True)
    parser.add_argument("--output-dir", "-o", default="data/calo/processed/")
    parser.add_argument("--energy-filter", "-e", type=float, default=None,
                        help="Select events at this incident energy (in GeV)")
    parser.add_argument("--max-events", "-n", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.input, "r") as f:
        showers = f["showers"][()]      # (N, 368) in MeV
        energies = f["incident_energies"][()]  # (N, 1) in MeV

    energies = energies.ravel()
    print(f"Loaded {len(showers)} showers, {showers.shape[1]} voxels")
    print(f"Energy range: {energies.min():.1f} - {energies.max():.1f} MeV")
    print(f"Unique energies: {np.unique(energies)}")

    if args.energy_filter is not None:
        # Convert GeV to MeV
        target_MeV = args.energy_filter * 1000
        mask = np.abs(energies - target_MeV) < 1.0
        if mask.sum() == 0:
            print(f"No events at {args.energy_filter} GeV, trying closest...")
            closest = np.unique(energies)[np.argmin(np.abs(np.unique(energies) - target_MeV))]
            mask = energies == closest
            print(f"Using energy {closest:.1f} MeV")
        showers = showers[mask]
        energies = energies[mask]
        print(f"After energy filter: {len(showers)} events")

    if args.max_events > 0 and len(showers) > args.max_events:
        showers = showers[:args.max_events]
        energies = energies[:args.max_events]

    # Extract HLV features for flowBDT training
    hlv = extract_hlv(showers, energies)
    print(f"HLV features shape: {hlv.shape}")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:>15s}: mean={hlv[:, i].mean():.6f} std={hlv[:, i].std():.6f}")

    # Save
    np.save(os.path.join(args.output_dir, "calo_hlv.npy"), hlv)
    np.save(os.path.join(args.output_dir, "calo_raw.npy"), showers)
    np.save(os.path.join(args.output_dir, "calo_energies.npy"), energies)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
