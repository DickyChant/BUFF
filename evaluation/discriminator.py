"""MLP classifier test for distinguishing real vs generated samples."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from BUFF.evaluation.bootstrap import bootstrap_metric


def train_discriminator(real, gen, hidden_layer_sizes=(128, 64), max_iter=200, seed=42):
    """Train an MLP to distinguish real from generated; return AUCs.

    Parameters
    ----------
    real, gen : array-like, shape (N, d)
        Real and generated feature arrays.

    Returns
    -------
    dict with auc_test, auc_train.
    """
    real, gen = np.asarray(real), np.asarray(gen)
    X = np.vstack([real, gen])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(gen))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)

    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]

    return {
        "auc_train": float(roc_auc_score(y_train, prob_train)),
        "auc_test": float(roc_auc_score(y_test, prob_test)),
    }


def discriminator_auc_with_errors(real, gen, n_bootstrap=20, seed=42):
    """Discriminator AUC with bootstrap uncertainty.

    Each bootstrap iteration re-trains the MLP on a resampled dataset.
    Uses fewer bootstrap iterations by default since each one involves
    training an MLP.

    Returns
    -------
    dict with auc_mean, auc_std.
    """
    rng = np.random.RandomState(seed)
    real, gen = np.asarray(real), np.asarray(gen)
    n_r, n_g = len(real), len(gen)
    aucs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx_r = rng.randint(0, n_r, size=n_r)
        idx_g = rng.randint(0, n_g, size=n_g)
        result = train_discriminator(real[idx_r], gen[idx_g], seed=seed + b)
        aucs[b] = result["auc_test"]
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }
