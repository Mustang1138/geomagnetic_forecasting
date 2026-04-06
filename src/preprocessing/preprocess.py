"""
Data preprocessing pipeline for geomagnetic forecasting.

Implements a unified preprocessing strategy for both baseline regression
models and temporal (LSTM/GRU) models, ensuring strict comparability.

Design principles:
- Chronological splitting only (no leakage)
- Identical cleaning for all models
- Scaling fitted on training data only
- 6-hourly resampling to match real-time inference cadence
- Per-model sequence arrays to honour each model's sequence_length

Rationale:
Time-series forecasting models are highly sensitive to data leakage
and inconsistent preprocessing. Enforcing a single pipeline ensures
fair model comparison and reproducibility
(Box et al., 2015; Cerqueira et al., 2020; Liemohn et al., 2021).

References:
- Box et al. (2015) - Time series analysis and forecasting methods
- Cerqueira et al. (2020) - Time series model evaluation best practices
- Cristoforetti et al. (2022) - Preprocessing importance in geomagnetic forecasting
- Hochreiter and Schmidhuber (1997) - LSTM architecture and training
- Papitashvili and King (2020) - OMNI2 data documentation
- Pedregosa et al. (2011) - scikit-learn preprocessing utilities
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.validators import validate_omni_dataframe
from src.features.derived_features import add_all_derived_features
from src.utils import load_config, ensure_dir, setup_logging

# Standardisation ensures zero-mean, unit-variance features, which is
# particularly important for gradient-based neural networks such as LSTMs.
# This prevents features with larger scales from dominating the learning
# process and improves convergence (Hochreiter and Schmidhuber, 1997;
# Paszke et al., 2019).

logger = setup_logging()


class DataPreprocessor:
    """
    Unified preprocessing for baseline and temporal ML models.

    This class implements a single, consistent preprocessing pipeline that
    produces data for both:
    1. Baseline models (tabular CSV format)
    2. Temporal models — LSTM and GRU (model-specific NumPy sequence arrays)

    Using a single preprocessing class ensures:
    - Identical data cleaning across all models
    - Identical scaling (fitted on training data only)
    - Consistent train/validation/test splits
    - No subtle experimental biases from pipeline divergence

    The pipeline resamples raw hourly OMNI2 data to 6-hourly averages,
    matching the cadence of the real-time DSCOVR inference pipeline
    (realtime_pipeline.py). This eliminates the train/inference distribution
    shift that would otherwise occur if models were calibrated on hourly
    patterns but received 6-hourly inputs at deployment.

    Attributes
    ----------
    FEATURE_COLS : list of str
        Solar wind input features used for prediction.
    TARGET_COL : str
        Target variable for regression (Storm Severity Index).
    config : dict
        Configuration dictionary loaded from YAML file.
    scaler_X : StandardScaler
        Fitted scaler for input features (prevents target leakage).
    scaler_y : StandardScaler
        Fitted scaler for target variable (enables inverse transformation).
    """

    # Input features used by ML models.
    # Bt is included as an explicit coupling/energy term alongside Bz.
    # dst is included as a direct input feature because it is the dominant
    # component of SSI (weight 0.30) and provides the models with explicit
    # access to the primary geomagnetic response signal. At prediction time,
    # the current Dst reading is genuinely available before the next
    # timestep's SSI is observed, so this does not constitute data leakage.
    FEATURE_COLS = ["bt", "bz_gsm", "speed", "density", "dst"]

    # Target variable: continuous Storm Severity Index
    TARGET_COL = "storm_severity_index"

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise the data preprocessor.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file.
            Default is "config.yaml".

        Notes
        -----
        Separate scalers are maintained for inputs (X) and target (y) to:
        1. Prevent target leakage into feature scaling
        2. Enable inverse transformation of predictions during evaluation
        3. Maintain independence between feature and target distributions

        This separation is a best practice in machine learning preprocessing
        (Pedregosa et al., 2011).
        """
        # Load configuration from YAML file
        self.config = load_config(config_path)

        # Scaler for input features (X).
        # Will be fitted on training data only to prevent information leakage.
        self.scaler_X = StandardScaler()

        # Scaler for target variable (y).
        # Kept separate to enable inverse transformation of model predictions
        # back to original scale for interpretable error metrics.
        self.scaler_y = StandardScaler()

    # Core cleaning

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward- and backward-filling.

        For time series data, forward/backward filling preserves temporal
        continuity without introducing synthetic trends. This is preferable
        to mean imputation, which can create artificial discontinuities in
        physical time series (Cristoforetti et al., 2022).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with potential missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled and remaining NaNs removed.

        Notes
        -----
        Filling is applied BEFORE 6-hourly resampling so that the
        aggregation operates on a complete hourly series and does not
        propagate NaNs into 6-hour bins.
        """
        # Ensure chronological ordering before filling
        df = df.sort_values("datetime").reset_index(drop=True)

        cols = self.FEATURE_COLS

        # Forward-fill followed by backward-fill handles gaps at both ends.
        df[cols] = df[cols].ffill().bfill()

        # Drop any remaining NaN rows (only if an entire column is missing)
        return df.dropna(subset=cols)

    def _resample_6hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample hourly OMNI2 data to 6-hourly averages.

        The real-time forecasting pipeline (realtime_pipeline.py) resamples
        DSCOVR observations to 6-hourly averages before model inference.
        Training data must share the same temporal cadence to avoid a
        train/inference distribution shift, which would cause models
        calibrated on hourly patterns to receive 6-hourly inputs at
        deployment — degrading forecast reliability
        (Cristoforetti et al., 2022).

        Averaging over 6-hour windows also reduces high-frequency noise
        that is not geophysically meaningful for storm-scale forecasting
        (Papitashvili and King, 2020).

        Parameters
        ----------
        df : pd.DataFrame
            Hourly DataFrame with a 'datetime' column and feature columns.

        Returns
        -------
        pd.DataFrame
            6-hourly resampled DataFrame, reset to a clean integer index.

        Notes
        -----
        Must be called AFTER missing-value filling and BEFORE outlier
        removal, so the aggregation operates on a complete hourly series.
        """
        df = (
            df.set_index("datetime")
            .resample("6h")
            .mean(numeric_only=True)
            .dropna(how="all")
            .reset_index()
        )
        logger.info("Resampled to 6-hourly cadence: %d rows.", len(df))
        return df

    def _remove_physical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove samples outside physically plausible limits.

        Physical outliers can arise from instrument errors, data transmission
        issues, or processing artefacts. Removing clearly unphysical values
        improves model robustness and prevents training instability
        (Liemohn et al., 2021).

        Thresholds are defined in config.yaml under physical_limits,
        supporting reproducibility and sensitivity analysis without
        modifying code (Martin, 2008).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with potential outliers.

        Returns
        -------
        pd.DataFrame
            DataFrame with physically implausible samples removed.
        """
        limits = self.config["physical_limits"]

        for col, (low, high) in limits.items():
            if col in df.columns:
                df = df[(df[col] >= low) & (df[col] <= high)]

        return df

    # Splitting & scaling

    def _split(
            self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronologically split data into train, validation, and test sets.

        For time series forecasting, chronological splitting is essential
        to prevent temporal leakage. Random splitting would allow the model
        to "peek into the future", invalidating forecasting performance
        estimates (Box et al., 2015; Cerqueira et al., 2020).

        Split proportions are read from config.yaml (training.test_split
        and training.validation_split). The remainder forms the training set.

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset sorted chronologically.

        Returns
        -------
        train, val, test : pd.DataFrame
            Chronological splits with earliest data in train and most
            recent data in test.
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

    def _scale(
            self,
            train: pd.DataFrame,
            val: pd.DataFrame,
            test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit scalers on training data only and apply to all splits.

        StandardScaler centres features to zero mean and unit variance.
        Scalers are fitted ONLY on the training split to prevent information
        leakage from validation/test sets (Pedregosa et al., 2011;
        Cerqueira et al., 2020).

        Separate scalers for X and y allow inverse transformation of
        predictions back to physical SSI units during evaluation.

        Parameters
        ----------
        train, val, test : pd.DataFrame
            Chronological splits.

        Returns
        -------
        train, val, test : pd.DataFrame
            Standardised splits (validation and test use training statistics).
        """
        # Fit on training data only
        self.scaler_X.fit(train[self.FEATURE_COLS])
        self.scaler_y.fit(train[[self.TARGET_COL]])

        # Transform all splits using training statistics
        for df in (train, val, test):
            df[self.FEATURE_COLS] = self.scaler_X.transform(df[self.FEATURE_COLS])
            df[self.TARGET_COL] = self.scaler_y.transform(df[[self.TARGET_COL]])

        return train, val, test

    # Sequence generation

    def _make_sequences(
            self, df: pd.DataFrame, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert tabular data into sliding window sequences for LSTM/GRU.

        Creates overlapping windows of length seq_len, each predicting the
        immediately following target value (sequence-to-one forecasting).

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed and scaled DataFrame.
        seq_len : int
            Length of input sequences (number of 6-hourly time steps).

        Returns
        -------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
        y : np.ndarray, shape (n_samples, 1)
        """
        features = df[self.FEATURE_COLS].values
        target = df[self.TARGET_COL].values

        X, y = [], []
        for i in range(len(df) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(target[i + seq_len])

        return np.array(X), np.array(y).reshape(-1, 1)

    # Public pipeline

    def run(
            self,
            input_csv: str = "data/raw/omni2_combined.csv",
            output_dir: str = "data/processed",
    ) -> Dict[str, object]:
        """
        Execute the full preprocessing pipeline and persist outputs.

        Pipeline stages:
        1.  Load raw OMNI2 CSV
        2.  Validate schema
        3.  Fill missing values (ffill/bfill)
        4.  Resample to 6-hourly averages (if config.data.resample_6h is true)
        5.  Remove physical outliers
        6.  Compute derived features (SSI, storm class, auroral latitude)
            in physical space — BEFORE scaling
        7.  Chronological split (train / val / test)
        8.  Fit StandardScalers on training data; transform all splits
        9.  Persist baseline CSVs for RF and LR
        10. Generate and persist per-model sequence arrays for LSTM and GRU
        11. Persist fitted scalers

        Parameters
        ----------
        input_csv : str
            Path to the combined raw OMNI2 CSV.
        output_dir : str
            Directory for all processed outputs.

        Returns
        -------
        dict
            Summary statistics describing the processed dataset.

        Output files
        ------------
        Baseline CSVs (for RF / LR):
            train_baseline.csv, val_baseline.csv, test_baseline.csv

        Per-model sequence arrays (for LSTM and GRU respectively):
            X_train_lstm.npy, y_train_lstm.npy
            X_val_lstm.npy,   y_val_lstm.npy
            X_test_lstm.npy,  y_test_lstm.npy
            X_train_gru.npy,  y_train_gru.npy
            X_val_gru.npy,    y_val_gru.npy
            X_test_gru.npy,   y_test_gru.npy

        Scalers:
            scaler_X.pkl, scaler_y.pkl
        """
        logger.info("Starting preprocessing pipeline")

        ensure_dir(output_dir)
        out = Path(output_dir)

        # Load
        df = pd.read_csv(input_csv, parse_dates=["datetime"])
        validate_omni_dataframe(df)

        # Clean
        df = self._handle_missing(df)

        # Resample to 6-hourly cadence when configured (default: True).
        # Must occur AFTER filling and BEFORE outlier removal.
        if self.config.get("data", {}).get("resample_6h", True):
            df = self._resample_6hourly(df)

        df = self._remove_physical_outliers(df)

        # Derived features (computed in physical space before scaling)
        df = add_all_derived_features(df)

        # Sanity check — no missing values should remain after imputation
        assert not df[self.FEATURE_COLS + [self.TARGET_COL]].isnull().any().any(), (
            "Missing values remain after imputation — check pipeline."
        )

        # Split
        train, val, test = self._split(df)
        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train), len(val), len(test),
        )

        # Scale (train-only fit)
        train, val, test = self._scale(train, val, test)

        # Persist baseline CSVs
        train.to_csv(out / "train_baseline.csv", index=False)
        val.to_csv(out / "val_baseline.csv", index=False)
        test.to_csv(out / "test_baseline.csv", index=False)
        logger.info("Baseline CSVs saved.")

        # Persist per-model sequence arrays
        # Each temporal model reads its own sequence_length from config.yaml.
        # Saving separate arrays corrects the previous behaviour where both
        # models consumed the LSTM sequence_length, inadvertently giving the
        # GRU a shorter temporal context than intended.
        for model_key in ("lstm", "gru"):
            seq_len = self.config["models"][model_key]["sequence_length"]

            X_tr, y_tr = self._make_sequences(train, seq_len)
            X_vl, y_vl = self._make_sequences(val, seq_len)
            X_te, y_te = self._make_sequences(test, seq_len)

            np.save(out / f"X_train_{model_key}.npy", X_tr)
            np.save(out / f"y_train_{model_key}.npy", y_tr)
            np.save(out / f"X_val_{model_key}.npy", X_vl)
            np.save(out / f"y_val_{model_key}.npy", y_vl)
            np.save(out / f"X_test_{model_key}.npy", X_te)
            np.save(out / f"y_test_{model_key}.npy", y_te)

            logger.info(
                "%s sequences: train=%d  val=%d  test=%d  (seq_len=%d)",
                model_key.upper(), len(X_tr), len(X_vl), len(X_te), seq_len,
            )

        # Persist scalers
        with open(out / "scaler_X.pkl", "wb") as fh:
            pickle.dump(self.scaler_X, fh)
        with open(out / "scaler_y.pkl", "wb") as fh:
            pickle.dump(self.scaler_y, fh)
        logger.info("Scalers saved.")

        # Summary
        lstm_seq = self.config["models"]["lstm"]["sequence_length"]
        gru_seq = self.config["models"]["gru"]["sequence_length"]

        summary = {
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "lstm_sequence_length": lstm_seq,
            "gru_sequence_length": gru_seq,
            "n_features": len(self.FEATURE_COLS),
            "target": self.TARGET_COL,
            "target_name": self.TARGET_COL,
            "feature_names": self.FEATURE_COLS.copy(),
        }

        logger.info("Preprocessing complete: %s", summary)
        return summary


def main():
    """
    Entry point for the preprocessing pipeline.

    Usage
    -----
    python -m src.preprocessing.preprocess

    Reads config.yaml, processes data/raw/omni2_combined.csv, and writes
    all outputs to data/processed/.
    """
    DataPreprocessor().run()


if __name__ == "__main__":
    main()
