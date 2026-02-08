import numpy as np

from src.models.persistence import persistence_forecast


def test_persistence_forecast_basic():
    """
    Persistence baseline should predict y[t] as y[t+1],
    dropping the first timestep.
    """
    y = np.array([0.1, 0.2, 0.4, 0.3])

    y_true, y_pred = persistence_forecast(y)

    # Expected behaviour:
    # y_pred[t] = y[t]
    # y_true[t] = y[t+1]
    np.testing.assert_allclose(y_pred, np.array([0.1, 0.2, 0.4]))
    np.testing.assert_allclose(y_true, np.array([0.2, 0.4, 0.3]))


def test_persistence_forecast_length():
    """
    Output arrays should be length T-1.
    """
    y = np.arange(10)

    y_true, y_pred = persistence_forecast(y)

    assert len(y_true) == len(y) - 1
    assert len(y_pred) == len(y) - 1


def test_persistence_forecast_requires_min_length():
    """
    Persistence baseline requires at least 2 timesteps.
    """
    y = np.array([0.5])

    try:
        persistence_forecast(y)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for input length < 2")
