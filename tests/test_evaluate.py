"""
Unit tests for model evaluation utilities.

This test module validates the evaluation pipeline, ensuring that
metrics are computed correctly and diagnostic plots are generated
without errors.

The tests use synthetic prediction data to verify functionality
independently of actual model training, following software testing
best practices (Martin, 2008).

References:
- Martin (2008) - Clean code and testing principles
- Liemohn et al. (2021) - Evaluation metrics for space physics models
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluate import evaluate_baseline_models


def _make_fake_predictions(path: Path, n: int = 200):
    """
    Create a synthetic predictions CSV matching baseline output format.

    This helper function generates dummy prediction data that mimics
    the structure of actual model outputs, enabling unit tests to run
    without requiring trained models.

    Parameters
    ----------
    path : Path
        File path where the synthetic CSV will be saved.
    n : int, optional
        Number of prediction samples to generate.
        Default is 200.

    Notes
    -----
    The synthetic data consists of:
    - y_true: Linearly spaced values from 0.0 to 1.0 (ground truth SSI)
    - y_pred: Linearly spaced values from 0.05 to 0.95 (predictions)

    The slight offset between y_true and y_pred simulates prediction
    error whilst maintaining a strong correlation (R² ≈ 0.99).

    This structure enables tests to verify:
    1. Metrics can be computed without errors
    2. Files are read and parsed correctly
    3. Plots are generated successfully

    The predictions are not realistic (linear relationship, no noise),
    but realism is not required for testing code functionality
    (Martin, 2008).
    """
    # Generate synthetic predictions with slight offset from truth
    # This creates a nearly perfect linear relationship
    df = pd.DataFrame({
        "y_true": np.linspace(0.0, 1.0, n),  # Ground truth SSI
        "y_pred": np.linspace(0.05, 0.95, n),  # Predictions (slight offset)
    })

    # Save to CSV in the expected format
    df.to_csv(path, index=False)


def test_evaluate_baseline_models_runs(tmp_path):
    """
    Test that baseline evaluation pipeline executes without errors.

    This integration test validates the complete evaluation workflow:
    1. Prediction file discovery
    2. Metrics computation
    3. Plot generation
    4. Metrics CSV creation

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory for test outputs.
        This ensures tests don't interfere with actual project files and
        remain isolated (pytest built-in fixture).

    Assertions
    ----------
    The test verifies that:
    - Evaluation runs without raising exceptions
    - metrics_baselines.csv is created
    - Expected metric columns are present (rmse, mae, r2)
    - Expected model rows are present (linear_regression, random_forest)

    Notes
    -----
    This test focuses on pipeline correctness, not metric values.
    With synthetic data, the actual metric values are meaningless,
    but the test ensures the evaluation code executes correctly and
    produces all expected outputs (Martin, 2008).

    The test uses a temporary directory (tmp_path) to avoid polluting
    the project directory and ensure automatic cleanup after the test
    completes.

    The synthetic predictions are created with a specific structure
    that matches the output format of baseline_models.py, ensuring
    compatibility between training and evaluation pipelines.
    """
    # Create expected directory structure
    # Mimics the actual project layout used in production
    results_dir = tmp_path / "results"
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True)

    # Create fake prediction files for both baseline models
    # These files match the naming convention used by baseline_models.py
    _make_fake_predictions(
        predictions_dir / "linear_regression_test_predictions.csv"
    )
    _make_fake_predictions(
        predictions_dir / "random_forest_test_predictions.csv"
    )

    # Run evaluation pipeline
    # processed_dir is not currently used by evaluate_baseline_models,
    # but is included for API consistency
    evaluate_baseline_models(
        processed_dir=tmp_path / "processed",  # Not used but required
        results_dir=results_dir,
    )

    # Verify metrics file exists
    # The evaluation function should create this file
    metrics_path = results_dir / "metrics_baselines.csv"
    assert metrics_path.exists(), \
        "metrics_baselines.csv was not created"

    # Verify metrics file contents
    # Load and inspect the CSV to ensure it has the expected structure
    metrics = pd.read_csv(metrics_path, index_col=0)

    # Check that all expected metric columns are present
    # These are the standard regression metrics (Liemohn et al., 2021)
    assert "rmse" in metrics.columns, \
        "RMSE column missing from metrics"
    assert "mae" in metrics.columns, \
        "MAE column missing from metrics"
    assert "r2" in metrics.columns, \
        "R² column missing from metrics"

    # Check that both baseline models are represented
    # The index contains model names
    assert "linear_regression" in metrics.index, \
        "Linear Regression results missing from metrics"
    assert "random_forest" in metrics.index, \
        "Random Forest results missing from metrics"