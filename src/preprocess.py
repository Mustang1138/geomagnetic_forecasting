"""
Data preprocessing pipeline for geomagnetic forecasting.

Implements a unified preprocessing strategy for both baseline regression
models and temporal (LSTM) models, ensuring strict comparability.

Design principles:
- Chronological splitting only (no leakage)
- Identical cleaning for all models
- Scaling fitted on training data only
- Baseline and LSTM data derived from the same source

Rationale:
Time-series forecasting models are highly sensitive to data leakage
and inconsistent preprocessing. Enforcing a single pipeline ensures
fair model comparison and reproducibility
(Box et al., 2015; Cerqueira et al., 2020; Liemohn et al., 2021).
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, ensure_dir, setup_logging
from src.validators import validate_omni_dataframe

# Standardisation ensures zero-mean, unit-variance features, which is
# particularly important for gradient-based neural networks such as LSTMs
# (Hochreiter and Schmidhuber, 1997; Paszke et al., 2019).

logger = setup_logging()


class DataPreprocessor:
    """
    Unified preprocessing for baseline and temporal ML models.

    A single preprocessing class is used to ensure:
    - identical data cleaning
    - identical scaling
    - consistent train/validation/test splits

    This avoids subtle experimental bias caused by pipeline divergence
    (Cristoforetti et al., 2022).
    """

    FEATURE_COLS = ["bz_gsm", "speed", "density"]
    TARGET_COL = "dst"

    # Feature selection follows established solar wind → Dst prediction
    # literature (Abduallah et al., 2022; Zou et al., 2024).

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        # Separate scalers for inputs and target prevent target leakage
        # and allow inverse-transforming predictions later
        # (Pedregosa et al., 2011).
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    # Core cleaning steps (shared by all models)

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward- and backward-filling.

        Forward/backward filling preserves temporal continuity without
        introducing synthetic trends, which is preferable for physical
        time series compared to mean imputation
        (Cristoforetti et al., 2022).
        """
        df = df.sort_values("datetime").reset_index(drop=True)

        cols = self.FEATURE_COLS + [self.TARGET_COL]
        # Forward-fill followed by backward-fill ensures gaps at both
        # ends of the series are handled deterministically.

        df[cols] = df[cols].ffill().bfill()

        # Remaining NaNs indicate unrecoverable missing data and are dropped
        df = df.dropna(subset=cols)
        return df

    def _remove_physical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove samples outside physically plausible limits.

        Outlier thresholds are defined in config.yaml to allow
        experiment-level control without modifying code.

        Removing physically impossible values improves model robustness
        and prevents training instability
        (Liemohn et al., 2021).
        """
        limits = self.config["physical_limits"]

        for col, (low, high) in limits.items():
            if col in df.columns:
                df = df[(df[col] >= low) & (df[col] <= high)]

        return df

    # Splitting & scaling

    def _split(self,
               df: pd.DataFrame) -> Tuple[pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame]:
        """
        Chronologically split data into train, validation, and test sets.

        Random splitting is explicitly avoided to prevent temporal leakage,
        which would invalidate forecasting performance estimates
        (Box et al., 2015).
        """
        test_frac = self.config["training"]["test_split"]
        val_frac = self.config["training"]["validation_split"]

        n = len(df)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_test - n_val

        train = df.iloc[:n_train].copy()
        val = df.iloc[n_train:n_train + n_val].copy()
        test = df.iloc[n_train + n_val:].copy()

        return train, val, test

    def _scale(self,
               train: pd.DataFrame,
               val: pd.DataFrame,
               test: pd.DataFrame
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit scalers on training data only and apply to all splits.

        This prevents information leakage from validation/test sets
        into the training process
        (Pedregosa et al., 2011; Cerqueira et al., 2020).
        """
        self.scaler_X.fit(train[self.FEATURE_COLS])
        self.scaler_y.fit(train[[self.TARGET_COL]])

        for df in (train, val, test):
            df[self.FEATURE_COLS] = self.scaler_X.transform(
                df[self.FEATURE_COLS])
            df[self.TARGET_COL] = self.scaler_y.transform(
                df[[self.TARGET_COL]])

        return train, val, test

    # LSTM sequence generation

    # Note:
    # Sequence construction necessarily reduces the number of usable samples
    # by `sequence_length`. This affects only temporal models; baseline models
    # operate on the full tabular dataset. This behaviour is expected and
    # documented to ensure transparency in model comparison.

    def _make_sequences(self,
                        df: pd.DataFrame,
                        seq_len: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert tabular data into sliding window sequences for LSTM models.

        Each input sample consists of `seq_len` consecutive time steps,
        predicting the subsequent Dst value. This formulation aligns
        with standard sequence-to-one forecasting setups
        (Hochreiter and Schmidhuber, 1997; Abduallah et al., 2022).
        """

        X, y = [], []
        features = df[self.FEATURE_COLS].values
        target = df[self.TARGET_COL].values

        for i in range(len(df) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(target[i + seq_len])

        return np.array(X), np.array(y).reshape(-1, 1)

    # Public pipeline

    def run(self,
            input_csv: str = "data/raw/omni2_combined.csv",
            output_dir: str = "data/processed"
            ) -> Dict[str, int]:
        """
        Execute the full preprocessing pipeline and persist outputs.

        Outputs include:
        - CSV files for baseline models
        - NumPy arrays for LSTM training
        - Fitted scalers for reproducibility
        """

        logger.info("Starting preprocessing pipeline")

        ensure_dir(output_dir)
        out = Path(output_dir)

        df = pd.read_csv(input_csv, parse_dates=["datetime"])

        # Early validation ensures schema correctness before mutation
        validate_omni_dataframe(df)

        df = self._handle_missing(df)
        df = self._remove_physical_outliers(df)

        train, val, test = self._split(df)
        train, val, test = self._scale(train, val, test)

        # Persist baseline (tabular) datasets
        train.to_csv(out / "train_baseline.csv", index=False)
        val.to_csv(out / "val_baseline.csv", index=False)
        test.to_csv(out / "test_baseline.csv", index=False)

        # Persist LSTM-ready sequences
        seq_len = self.config["models"]["lstm"]["sequence_length"]

        X_train, y_train = self._make_sequences(train, seq_len)
        X_val, y_val = self._make_sequences(val, seq_len)
        X_test, y_test = self._make_sequences(test, seq_len)

        np.save(out / "X_train.npy", X_train)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "X_val.npy", X_val)
        np.save(out / "y_val.npy", y_val)
        np.save(out / "X_test.npy", X_test)
        np.save(out / "y_test.npy", y_test)

        # Persist scalers to enable inverse transformation of predictions
        # during evaluation and deployment.
        with open(out / "scaler_X.pkl", "wb") as f:
            pickle.dump(self.scaler_X, f)

        with open(out / "scaler_y.pkl", "wb") as f:
            pickle.dump(self.scaler_y, f)

        summary = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "sequence_length": seq_len,
            "n_features": len(self.FEATURE_COLS),
            "feature_names": self.FEATURE_COLS,
            "target_name": self.TARGET_COL,
        }

        logger.info(f"Preprocessing complete: {summary}")
        return summary


def main():
    pre = DataPreprocessor()
    summary = pre.run()

    print("\nPreprocessing complete")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
