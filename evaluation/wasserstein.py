"""Wasserstein-1 distance with bootstrap uncertainties."""

import numpy as np
from scipy.stats import wasserstein_distance

from BUFF.evaluation.bootstrap import bootstrap_metric, bootstrap_truth_baseline


def _w1_1d(real, gen):
    """1-D Wasserstein-1 distance (Earth Mover's Distance)."""
    return wasserstein_distance(real.ravel(), gen.ravel())


def w1_with_errors(real, gen, n_bootstrap=100, seed=42):
    """Wasserstein-1 distance with bootstrap errors and truth baseline.

    Parameters
    ----------
    real, gen : array-like, shape (N,)
        1-D samples.

    Returns
    -------
    dict with w1_mean, w1_std, truth_mean, truth_std.
    """
    result = bootstrap_metric(real, gen, _w1_1d, n_bootstrap=n_bootstrap, seed=seed)
    truth = bootstrap_truth_baseline(real, _w1_1d, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "w1_mean": result["mean"],
        "w1_std": result["std"],
        "truth_mean": truth["mean"],
        "truth_std": truth["std"],
    }
