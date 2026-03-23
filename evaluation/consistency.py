"""Consistency checks for derived jet substructure quantities."""

import numpy as np
from scipy.stats import wasserstein_distance


def tau21_consistency_check(gen_tau1, gen_tau2, gen_tau21):
    """Check consistency between generated tau21 and tau2/tau1.

    Compares the directly generated tau21 with the ratio computed
    from generated tau1 and tau2.  A well-trained model should produce
    small W1 distance between these two.

    Returns
    -------
    dict with w1 (Wasserstein-1 distance) and derived_tau21 array.
    """
    gen_tau1 = np.asarray(gen_tau1).ravel()
    gen_tau2 = np.asarray(gen_tau2).ravel()
    gen_tau21 = np.asarray(gen_tau21).ravel()

    # Avoid division by zero
    mask = gen_tau1 > 0
    derived = gen_tau2[mask] / gen_tau1[mask]
    direct = gen_tau21[mask]

    w1 = wasserstein_distance(derived, direct)
    return {"w1": float(w1), "derived_tau21": derived, "direct_tau21": direct}


def tau32_consistency_check(gen_tau2, gen_tau3, gen_tau32):
    """Same as tau21_consistency_check but for tau32 = tau3/tau2."""
    gen_tau2 = np.asarray(gen_tau2).ravel()
    gen_tau3 = np.asarray(gen_tau3).ravel()
    gen_tau32 = np.asarray(gen_tau32).ravel()

    mask = gen_tau2 > 0
    derived = gen_tau3[mask] / gen_tau2[mask]
    direct = gen_tau32[mask]

    w1 = wasserstein_distance(derived, direct)
    return {"w1": float(w1), "derived_tau32": derived, "direct_tau32": direct}
