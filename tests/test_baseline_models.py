"""
Unit tests for baseline regression models.

This test module validates the baseline model training pipeline,
focusing on pipeline correctness, artefact persistence, and data
compatibility rather than predictive skill.

The tests use synthetic dummy data to ensure reproducibility and
independence from external data sources, following best practices
for software testing (Martin, 2008).

References:
- Martin (2008) - Clean code and testing principles
- Pedregosa et al. (2011) - scikit-learn testing patterns
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.baseline_models import BaselineTrainer


def _make_dummy_dataset(path: Path, n: int = 50):
    """
    Create a synthetic dataset for testing baseline models.

    This function generates random data that mimics the structure of
    preprocessed baseline datasets, enabling unit tests to run without
    requiring actual OMNI data or the full preprocessing pipeline.

    Parameters
    ----------
    path : Path
        File path where the synthetic CSV will be saved.
    n : int, optional
        Number of samples to generate.
        Default is 50.

    Notes
    -----
    The synthetic data includes:
    - bz_gsm: Random standard normal values (mimicking scaled IMF Bz)
    - speed: Uniform random values in [300, 800] (mimicking solar wind speed)
    - density: Uniform random values in [1, 30] (mimicking proton density)
    - storm_severity_index: Uniform random values in [0, 1] (mimicking SSI)

    These ranges loosely reflect the physical parameters but are not
    intended to be realistic. The purpose is to test code functionality,
    not scientific validity.

    A dummy scaler_y.pkl is also written to the parent directory on the
    first call, mimicking the artefact produced by the preprocessing
    pipeline.  The scaler is fitted on the generated SSI column so that
    inverse_transform calls in baseline_models.py succeed during tests.
    """
    # Generate synthetic data covering all five feature columns used by
    # BaselineTrainer.FEATURE_COLS: bt, bz_gsm, speed, density, dst.
    df = pd.DataFrame({
        "bt": np.random.uniform(0, 30, n),              # IMF magnitude (nT)
        "bz_gsm": np.random.randn(n),                   # IMF Bz (scaled)
        "speed": np.random.uniform(300, 800, n),        # Solar wind speed (km/s)
        "density": np.random.uniform(1, 30, n),         # Proton density (cm⁻³)
        "dst": np.random.uniform(-100, 10, n),          # Dst index (nT)
        "storm_severity_index": np.random.rand(n),      # SSI in [0, 1]
    })

    # Save to CSV (mimics preprocessed baseline data format)
    df.to_csv(path, index=False)

    # Write a dummy scaler_y.pkl fitted on this dataset's SSI column.
    # baseline_models.py expects this file to be present in the processed
    # directory so it can inverse-transform scaled predictions before saving.
    # We only write it once (when train_baseline.csv is created) to avoid
    # overwriting with a differently-fitted scaler from the test-set call.
    scaler_path = path.parent / "scaler_y.pkl"
    if not scaler_path.exists():
        scaler_y = StandardScaler()
        scaler_y.fit(df[["storm_severity_index"]])
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_y, f)


def test_baseline_models_train_and_predict(tmp_path):
    """
    Test that baseline models train successfully and produce valid outputs.

    This integration test validates the complete baseline training pipeline:
    1. Model instantiation
    2. Training on synthetic data
    3. Prediction generation
    4. Metrics computation
    5. Artefact persistence (models and predictions)

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory for test outputs.
        This ensures tests don't pollute the project directory and remain
        isolated (pytest built-in fixture).

    Assertions
    ----------
    The test verifies:
    - Both models (Linear Regression, Random Forest) are trained
    - All expected metrics (RMSE, MAE, R²) are computed
    - Trained models are serialised and saved
    - Test predictions are saved to CSV files

    Notes
    -----
    This test focuses on pipeline correctness, not model performance.
    With random synthetic data, metrics values are meaningless, but the
    test ensures that the training pipeline executes without errors and
    produces all expected outputs (Martin, 2008).

    The use of a temporary directory (tmp_path) ensures test isolation
    and automatic cleanup, following pytest best practices.
    """
    # Set up temporary directory structure
    processed = tmp_path / "processed"
    results = tmp_path / "outputs"

    processed.mkdir()
    results.mkdir()

    # Create synthetic training and test datasets
    # Using larger training set (n=100) than test set (n=40) mimics
    # typical train/test split ratios
    _make_dummy_dataset(processed / "train_baseline.csv", n=100)
    _make_dummy_dataset(processed / "test_baseline.csv", n=40)

    # Instantiate trainer (uses default config.yaml)
    trainer = BaselineTrainer()

    # Run training pipeline on synthetic data
    metrics = trainer.run(
        processed_dir=str(processed),
        output_dir=str(results),
    )

    # Validate metrics structure
    # Ensure both models are present in outputs
    assert "linear_regression" in metrics, \
        "Linear Regression outputs missing from output"
    assert "random_forest" in metrics, \
        "Random Forest outputs missing from output"

    # Verify all expected metrics are computed for each model
    for model_name, model_metrics in metrics.items():
        assert "rmse" in model_metrics, \
            f"RMSE metric missing for {model_name}"
        assert "mae" in model_metrics, \
            f"MAE metric missing for {model_name}"
        assert "r2" in model_metrics, \
            f"R² metric missing for {model_name}"

    # Check persisted artefacts
    # Models should be saved as pickle files
    model_dir = results / "models"
    pred_dir = results / "predictions"

    # Verify model files exist
    assert (model_dir / "linear_regression.pkl").exists(), \
        "Linear Regression model file not found"
    assert (model_dir / "random_forest.pkl").exists(), \
        "Random Forest model file not found"

    # Verify prediction CSV files exist
    assert (pred_dir / "linear_regression_test_predictions.csv").exists(), \
        "Linear Regression predictions file not found"
    assert (pred_dir / "random_forest_test_predictions.csv").exists(), \
        "Random Forest predictions file not found"
