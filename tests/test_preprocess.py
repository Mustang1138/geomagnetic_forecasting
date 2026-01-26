import numpy as np
import pandas as pd

from src.preprocess import DataPreprocessor


def test_preprocess_shapes(tmp_path):
    """
    Integration-style test validating end-to-end preprocessing
    output shapes for LSTM models.
    """

    # Create minimal synthetic dataset
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=2000, freq="h"),
        "bz_gsm": np.random.normal(0, 5, 2000),
        "speed": np.random.normal(400, 50, 2000),
        "density": np.random.normal(5, 2, 2000),
        "dst": np.random.normal(-20, 10, 2000),
    })

    csv_path = tmp_path / "fake_omni.csv"
    df.to_csv(csv_path, index=False)

    pre = DataPreprocessor()
    summary = pre.run(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "processed"),
    )

    assert summary["train_samples"] > 0
    assert summary["val_samples"] > 0
    assert summary["test_samples"] > 0

    X = np.load(tmp_path / "processed" / "X_train.npy")
    y = np.load(tmp_path / "processed" / "y_train.npy")

    assert X.ndim == 3  # (samples, timesteps, features)
    assert y.ndim == 2
