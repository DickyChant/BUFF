"""Separation power (triangular discriminator) with bootstrap uncertainties."""

import numpy as np

from BUFF.evaluation.bootstrap import bootstrap_metric, bootstrap_truth_baseline


def _separation_power(real, gen, n_bins=50):
    """Separation power (triangular discriminator) between two 1-D samples.

    Sep = (1/2) Σ (h_r - h_g)^2 / (h_r + h_g)

    where h_r, h_g are normalised histogram bin counts.
    """
    real, gen = real.ravel(), gen.ravel()
    lo = min(real.min(), gen.min())
    hi = max(real.max(), gen.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    h_r, _ = np.histogram(real, bins=bins, density=False)
    h_g, _ = np.histogram(gen, bins=bins, density=False)
    h_r = h_r / h_r.sum()
    h_g = h_g / h_g.sum()
    denom = h_r + h_g
    mask = denom > 0
    return 0.5 * np.sum((h_r[mask] - h_g[mask]) ** 2 / denom[mask])


def sep_power_with_errors(real, gen, n_bootstrap=100, n_bins=50, seed=42):
    """Separation power with bootstrap errors and truth baseline.

    Returns
    -------
    dict with sep_mean, sep_std, truth_mean, truth_std.
    """

    def _metric(r, g):
        return _separation_power(r, g, n_bins=n_bins)

    result = bootstrap_metric(real, gen, _metric, n_bootstrap=n_bootstrap, seed=seed)
    truth = bootstrap_truth_baseline(real, _metric, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "sep_mean": result["mean"],
        "sep_std": result["std"],
        "truth_mean": truth["mean"],
        "truth_std": truth["std"],
    }
