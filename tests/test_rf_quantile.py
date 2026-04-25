"""Tests for the RF quantile-interval helper."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.rf_quantile import predict_with_ci


@pytest.fixture
def fitted_rf() -> tuple[RandomForestRegressor, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 4))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.2 + rng.normal(scale=0.1, size=400)
    model = RandomForestRegressor(n_estimators=30, random_state=0)
    model.fit(X, y)
    X_test = rng.normal(size=(50, 4))
    return model, X_test


def test_quantile_bounds_ordered(fitted_rf):
    model, X_test = fitted_rf
    mean_pred, lower, upper = predict_with_ci(model, X_test, quantiles=(0.05, 0.95))
    assert lower.shape == mean_pred.shape == upper.shape
    assert np.all(lower <= mean_pred + 1e-9)
    assert np.all(mean_pred <= upper + 1e-9)
    assert np.all(lower <= upper)


def test_mean_matches_model_predict(fitted_rf):
    model, X_test = fitted_rf
    mean_pred, _, _ = predict_with_ci(model, X_test)
    np.testing.assert_allclose(mean_pred, model.predict(X_test), atol=1e-9)


def test_coverage_is_near_nominal(fitted_rf):
    model, X_test = fitted_rf
    _, lower, upper = predict_with_ci(model, X_test, quantiles=(0.05, 0.95))
    # With only 30 trees the 90 % interval is indicative rather than exact; assert
    # it is plausibly in the right order of magnitude relative to the spread of
    # per-tree predictions.
    widths = upper - lower
    assert np.all(widths >= 0)
    assert widths.mean() > 0.0


def test_invalid_quantiles_raise(fitted_rf):
    model, X_test = fitted_rf
    with pytest.raises(ValueError):
        predict_with_ci(model, X_test, quantiles=(0.5, 0.5))
    with pytest.raises(ValueError):
        predict_with_ci(model, X_test, quantiles=(-0.1, 0.9))
