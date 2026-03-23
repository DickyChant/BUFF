"""Bootstrap uncertainty estimation for arbitrary metrics."""

import numpy as np


def bootstrap_metric(real, gen, metric_fn, n_bootstrap=100, seed=42):
    """Compute a metric with bootstrap uncertainty.

    Parameters
    ----------
    real, gen : array-like, shape (N,) or (N, d)
        Real and generated samples (matched or unmatched).
    metric_fn : callable(real, gen) → float
        Scalar metric to evaluate.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: mean, std, values (array of bootstrap replicates).
    """
    rng = np.random.RandomState(seed)
    real = np.asarray(real)
    gen = np.asarray(gen)
    n_real, n_gen = len(real), len(gen)
    values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx_r = rng.randint(0, n_real, size=n_real)
        idx_g = rng.randint(0, n_gen, size=n_gen)
        values[b] = metric_fn(real[idx_r], gen[idx_g])
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}


def bootstrap_truth_baseline(data, metric_fn, n_bootstrap=100, seed=42):
    """Finite-sample floor: split data in half and evaluate metric.

    This gives the expected metric value even for a perfect generator,
    due to finite statistics.
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)
    values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        perm = rng.permutation(n)
        half = n // 2
        a, b_split = data[perm[:half]], data[perm[half: half * 2]]
        values[b] = metric_fn(a, b_split)
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
