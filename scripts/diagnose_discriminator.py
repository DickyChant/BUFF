"""Diagnose what the discriminator is picking up on."""
import numpy as np
from BUFF.evaluation.discriminator import train_discriminator
from BUFF.runner.train_and_sample import JETNET_INDEPENDENT_INDICES

real = np.load("data/jetnet/t_hlv12.npy")
gen = np.load("results/jetnet_highlevel_v3/generated_samples.npy")

features = ["d12","d2","mass","pt","tau1","tau2","tau3","tau21","tau32","ecf2","ecf3","d2_obs"]

print("=== Baseline ===")
r = train_discriminator(real, gen)
print(f"AUC test: {r['auc_test']:.4f}")

# Test 1: Remove samples at exact boundaries
X_min = real.min(axis=0)
X_max = real.max(axis=0)
at_boundary = np.zeros(len(gen), dtype=bool)
for k in range(gen.shape[1]):
    at_boundary |= (gen[:, k] == X_min[k]) | (gen[:, k] == X_max[k])
print(f"\nTotal at boundary: {at_boundary.sum()} ({at_boundary.sum()/len(gen)*100:.2f}%)")
gen_noclip = gen[~at_boundary]

rng = np.random.RandomState(42)
real_sub = real[rng.choice(len(real), len(gen_noclip), replace=False)]

print("\n=== Test 1: Remove boundary-clipped samples ===")
r = train_discriminator(real_sub, gen_noclip)
print(f"AUC test: {r['auc_test']:.4f}")

# Test 2: Add small noise to break boundary spikes
gen_noisy = gen.copy()
noise = np.random.normal(0, 1e-4, gen.shape)
gen_noisy = gen_noisy + noise

print("\n=== Test 2: Add tiny noise to generated (break boundary spikes) ===")
r = train_discriminator(real, gen_noisy)
print(f"AUC test: {r['auc_test']:.4f}")

# Test 3: Discriminator on subset of features (exclude tau3)
tau3_idx = 6
feats_notau3 = [i for i in range(12) if i != tau3_idx]
print(f"\n=== Test 3: Exclude tau3 (feature {tau3_idx}) ===")
r = train_discriminator(real[:, feats_notau3], gen[:, feats_notau3])
print(f"AUC test: {r['auc_test']:.4f}")

# Test 4: Only independent features (9), no derived
print(f"\n=== Test 4: Only 9 independent features ===")
r = train_discriminator(real[:, JETNET_INDEPENDENT_INDICES], gen[:, JETNET_INDEPENDENT_INDICES])
print(f"AUC test: {r['auc_test']:.4f}")

# Test 5: Per-feature discriminators (which feature is most distinguishable?)
print("\n=== Test 5: Per-feature discriminator AUC ===")
for i, name in enumerate(features):
    r = train_discriminator(real[:, i:i+1], gen[:, i:i+1])
    print(f"  {name:6s}: AUC = {r['auc_test']:.4f}")

# Test 6: Pairs of features
print("\n=== Test 6: Worst feature pairs ===")
worst_pairs = [(6, 10), (3, 6), (6, 9), (4, 6), (4, 5)]  # tau3-related
for i, j in worst_pairs:
    idx = [i, j]
    r = train_discriminator(real[:, idx], gen[:, idx])
    print(f"  {features[i]:6s} x {features[j]:6s}: AUC = {r['auc_test']:.4f}")
