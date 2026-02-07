import pandas as pd

from src.evaluation.validators import (
    validate_omni_dataframe,
    validate_preprocessed_data,
)


def test_validate_omni_dataframe_basic():
    """
    Sanity test ensuring valid OMNI-like data passes schema
    and continuity validation.
    """
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=10, freq="h"),
        "bz_gsm": [0.0] * 10,
        "bt": [6.0] * 10,
        "speed": [400.0] * 10,
        "density": [5.0] * 10,
        "dst": [-10.0] * 10,
    })

    result = validate_omni_dataframe(df)

    assert result["total_records"] == 10
    assert result["date_continuity"]["num_gaps"] == 0


def test_validate_preprocessed_data_passes():
    """
    Ensure preprocessing summary meeting minimum thresholds
    does not raise validation errors.
    """
    summary = {
        "train_samples": 2000,
        "val_samples": 500,
        "test_samples": 500,
        "sequence_length": 24,
        "n_features": 5,
        "feature_names": ["bz_gsm", "bt", "speed", "density", "dst"],
        "target_name": "storm_severity_index",
    }

    validate_preprocessed_data(summary)  # should not raise
