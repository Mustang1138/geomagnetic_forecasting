"""
Unit tests for baseline regression models.

Focuses on pipeline correctness, artefact persistence,
and data compatibility rather than predictive skill.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline_models import BaselineTrainer


def _make_fake_baseline_csv(path: Path, n: int = 500, seed: int = 42) -> None:
    """
    Create a minimal, deterministic baseline dataset for testing.

    The data distribution is not physically meaningful; it is designed
    solely to validate pipeline behaviour and output shapes.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "bz_gsm": rng.normal(0, 5, n),
        "speed": rng.normal(400, 50, n),
        "density": rng.normal(5, 2, n),
        "dst": rng.normal(-20, 10, n),
    })

    df.to_csv(path, index=False)


@pytest.fixture
def baseline_trainer(tmp_path):
    """
    Provide a BaselineTrainer instance wired to a temporary
    synthetic preprocessing environment.
    """
    processed = tmp_path / "processed"
    processed.mkdir()

    for split in ("train", "val", "test"):
        _make_fake_baseline_csv(processed / f"{split}_baseline.csv")

    trainer = BaselineTrainer()
    trainer.config["data"]["processed_dir"] = str(processed)
    trainer.output_dir = tmp_path / "results"

    return trainer


def test_linear_regression_training(baseline_trainer):
    """
    Verify that linear regression trains successfully and produces
    correctly shaped predictions for all dataset splits.
    """
    preds = baseline_trainer.train_linear_regression()

    for split, arr in preds.items():
        assert arr.ndim == 2
        assert arr.shape[1] == 1
        assert arr.shape[0] > 0


def test_random_forest_training(baseline_trainer):
    """
    Verify that random forest training runs end-to-end and produces
    consistent prediction outputs.
    """
    preds = baseline_trainer.train_random_forest()

    for split, arr in preds.items():
        assert arr.ndim == 2
        assert arr.shape[1] == 1


def test_baseline_models_train_and_save(tmp_path, monkeypatch):
    """
    End-to-end validation test for baseline model training.

    This test verifies that:
    - Preprocessed baseline datasets are correctly consumed
    - Linear Regression and Random Forest models train without error
    - Predictions are generated for train/validation/test splits
    - Model artefacts and prediction outputs are persisted to disk

    The test intentionally avoids validating predictive performance,
    as baseline accuracy is evaluated separately during the analysis
    phase of the project.
    """

    # Arrange: Create isolated synthetic environment

    # Temporary directories emulate the expected project layout
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results" / "baselines"

    processed_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    # Generate synthetic baseline CSVs for each split
    # Dataset sizes exceed minimum thresholds used elsewhere in the project
    _make_fake_baseline_csv(processed_dir / "train_baseline.csv", n=1000)
    _make_fake_baseline_csv(processed_dir / "val_baseline.csv", n=300)
    _make_fake_baseline_csv(processed_dir / "test_baseline.csv", n=300)

    # Monkeypatch configuration loading to avoid dependence on config.yaml
    # This ensures tests are deterministic and self-contained
    def fake_load_config(_=None):
        return {
            "data": {
                "processed_dir": str(processed_dir),
            },
            "models": {
                "baseline": {
                    "random_forest": {
                        "n_estimators": 10,  # Reduced for fast test execution
                        "max_depth": 5,
                        "random_state": 42,
                    }
                }
            },
        }

    monkeypatch.setattr(
        "src.baseline_models.load_config",
        fake_load_config,
    )

    # Act: Train baseline models

    trainer = BaselineTrainer()

    # Override output directory to keep test artefacts isolated
    trainer.output_dir = results_dir

    lr_preds = trainer.train_linear_regression()
    rf_preds = trainer.train_random_forest()

    # Assert: Validate outputs and artefacts

    # Verify prediction dictionaries contain all expected splits
    for preds in (lr_preds, rf_preds):
        assert set(preds.keys()) == {"train", "val", "test"}

        # Predictions should be column vectors: (n_samples, 1)
        assert preds["train"].ndim == 2
        assert preds["val"].ndim == 2
        assert preds["test"].ndim == 2

    # Linear Regression artefacts
    lr_dir = results_dir / "linear_regression"
    assert (lr_dir / "model.pkl").exists()
    assert (lr_dir / "y_pred_train.npy").exists()
    assert (lr_dir / "y_pred_val.npy").exists()
    assert (lr_dir / "y_pred_test.npy").exists()

    # Random Forest artefacts
    rf_dir = results_dir / "random_forest"
    assert (rf_dir / "model.pkl").exists()
    assert (rf_dir / "y_pred_train.npy").exists()
    assert (rf_dir / "y_pred_val.npy").exists()
    assert (rf_dir / "y_pred_test.npy").exists()
