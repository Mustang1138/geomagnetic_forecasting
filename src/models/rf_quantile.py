"""Quantile prediction intervals from a trained Random Forest regressor.

Aggregates per-tree predictions from ``RandomForestRegressor.estimators_`` to
produce empirical quantile bounds around each prediction without retraining.
This is a standard operationalisation of quantile regression for tree
ensembles (Meinshausen, 2006) that exploits the natural diversity across
bootstrap-sampled trees as a proxy for the predictive distribution.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def predict_with_ci(
    model: RandomForestRegressor,
    X: np.ndarray,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict with a central estimate and empirical quantile bounds.

    Parameters
    ----------
    model
        A fitted ``RandomForestRegressor`` whose ``estimators_`` attribute
        provides per-tree access.
    X
        Feature matrix of shape (n_samples, n_features).
    quantiles
        ``(lower, upper)`` quantile pair in [0, 1]. Defaults to the 90 per cent
        central interval.

    Returns
    -------
    (mean_pred, lower, upper)
        Arrays of length ``n_samples``. ``mean_pred`` equals ``model.predict(X)``
        by construction; ``lower`` and ``upper`` are the requested quantiles of
        the per-tree prediction distribution.
    """
    q_lo, q_hi = quantiles
    if not 0.0 <= q_lo < q_hi <= 1.0:
        raise ValueError(f"Invalid quantile pair: ({q_lo}, {q_hi})")

    # Shape: (n_trees, n_samples)
    tree_preds = np.stack(
        [tree.predict(X) for tree in model.estimators_], axis=0
    )
    mean_pred = tree_preds.mean(axis=0)
    lower = np.quantile(tree_preds, q_lo, axis=0)
    upper = np.quantile(tree_preds, q_hi, axis=0)
    return mean_pred, lower, upper
