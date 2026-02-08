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

References:
- Box et al. (2015) - Time series analysis and forecasting methods
- Cerqueira et al. (2020) - Time series model evaluation best practices
- Hochreiter and Schmidhuber (1997) - LSTM architecture and training
- Pedregosa et al. (2011) - scikit-learn preprocessing utilities
- Cristoforetti et al. (2022) - Preprocessing importance in geomagnetic forecasting
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
    2. LSTM models (sequence arrays in NumPy format)

    Using a single preprocessing class ensures:
    - Identical data cleaning across all models
    - Identical scaling (fitted on training data only)
    - Consistent train/validation/test splits
    - No subtle experimental biases from pipeline divergence

    This approach is critical for fair model comparison and follows best
    practices for time series forecasting experiments (Cristoforetti et al., 2022;
    Cerqueira et al., 2020).

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
    FEATURE_COLS = ["bt", "bz_gsm", "speed", "density"]

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

        # Separate scalers for inputs and target prevent target leakage
        # and allow inverse-transforming predictions later (Pedregosa et al., 2011).

        # Scaler for input features (X)
        # Will be fitted on training data only to prevent information leakage
        self.scaler_X = StandardScaler()

        # Scaler for target variable (y)
        # Kept separate to enable inverse transformation of model predictions
        # back to original scale for interpretable error metrics
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
        The filling strategy:
        1. Sort by datetime to ensure temporal ordering
        2. Forward-fill (propagate last valid observation forward)
        3. Backward-fill (propagate next valid observation backward)
        4. Drop any remaining NaNs (e.g., if entire series is missing)

        This approach handles gaps at both ends of the series deterministically
        whilst preserving the temporal structure of the data.
        """
        # Ensure chronological ordering before filling
        # This is critical for meaningful forward/backward propagation
        df = df.sort_values("datetime").reset_index(drop=True)

        # Specify columns to fill
        # Include both features and Dst (needed for derived feature computation)
        cols = self.FEATURE_COLS + ["dst"]

        # Forward-fill followed by backward-fill ensures gaps at both
        # ends of the series are handled deterministically.
        # ffill(): propagate last valid value forward
        # bfill(): propagate next valid value backward
        df[cols] = df[cols].ffill().bfill()

        # Drop any remaining samples with NaN values
        # These would only remain if an entire column is missing
        return df.dropna(subset=cols)

    def _remove_physical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove samples outside physically plausible limits.

        Physical outliers can arise from instrument errors, data transmission
        issues, or processing artefacts. Removing clearly unphysical values
        improves model robustness and prevents training instability
        (Liemohn et al., 2021).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with potential outliers.

        Returns
        -------
        pd.DataFrame
            DataFrame with physically implausible samples removed.

        Notes
        -----
        Outlier thresholds are defined in config.yaml to allow experiment-level
        control without modifying code. This design supports reproducibility
        and facilitates sensitivity analysis on outlier handling strategies.

        Thresholds are based on:
        - OMNI data documentation (Papitashvili and King, 2020)
        - Physical constraints on solar wind parameters
        - Historical extreme event observations

        Samples are removed (not clipped) to avoid introducing artificial
        values at the boundaries.
        """
        # Load physical limits from configuration
        limits = self.config["physical_limits"]

        # Apply limits to each configured parameter
        for col, (low, high) in limits.items():
            if col in df.columns:
                # Remove samples outside [low, high] range
                # This is a row-wise filter, so samples violating any limit are removed
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

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset sorted chronologically.

        Returns
        -------
        train : pd.DataFrame
            Training set (earliest data).
        val : pd.DataFrame
            Validation set (middle period).
        test : pd.DataFrame
            Test set (most recent data, representing future predictions).

        Notes
        -----
        Split proportions are defined in config.yaml. Typical values:
        - Train: 70% (earliest data for model fitting)
        - Validation: 15% (middle data for hyperparameter tuning/early stopping)
        - Test: 15% (most recent data for final evaluation)

        The validation set is used by LSTM models for early stopping but is
        intentionally not used by baseline models, which employ fixed
        hyperparameters.

        Random splitting is explicitly avoided to prevent temporal leakage,
        which would invalidate forecasting performance estimates (Box et al., 2015).
        """
        # Extract split fractions from configuration
        test_frac = self.config["training"]["test_split"]
        val_frac = self.config["training"]["validation_split"]

        # Calculate split indices
        n = len(df)
        n_test = int(n * test_frac)  # Most recent data
        n_val = int(n * val_frac)  # Middle period
        n_train = n - n_test - n_val  # Earliest data

        # Perform chronological split
        # Earlier indices = earlier in time
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

        StandardScaler centres features to zero mean and unit variance,
        which is particularly important for:
        1. Gradient-based optimisation (LSTMs)
        2. Preventing features with larger scales from dominating
        3. Improving numerical stability

        Critically, scalers are fitted ONLY on training data to prevent
        information leakage from validation/test sets into the training
        process (Pedregosa et al., 2011; Cerqueira et al., 2020).

        Parameters
        ----------
        train : pd.DataFrame
            Training set (used to fit scalers).
        val : pd.DataFrame
            Validation set (scalers applied only).
        test : pd.DataFrame
            Test set (scalers applied only).

        Returns
        -------
        train : pd.DataFrame
            Scaled training set.
        val : pd.DataFrame
            Scaled validation set (using training statistics).
        test : pd.DataFrame
            Scaled test set (using training statistics).

        Notes
        -----
        Standardisation formula:
            z = (x - μ) / σ
        where μ and σ are computed from the training set only.

        This prevents information leakage: validation and test sets are
        transformed using training statistics, simulating a realistic
        deployment scenario where future data statistics are unknown.

        Separate scalers for X and y allow inverse transformation of
        predictions back to original scale during evaluation.
        """
        # Fit input feature scaler on training data only
        # This computes mean and standard deviation from training set
        self.scaler_X.fit(train[self.FEATURE_COLS])

        # Fit target scaler on training data only
        # Kept separate to enable inverse transformation of predictions
        self.scaler_y.fit(train[[self.TARGET_COL]])

        # Apply fitted scalers to all splits
        # Transform (but do not refit) validation and test sets
        for df in (train, val, test):
            # Transform features using training statistics
            df[self.FEATURE_COLS] = self.scaler_X.transform(
                df[self.FEATURE_COLS])

            # Transform target using training statistics
            df[self.TARGET_COL] = self.scaler_y.transform(
                df[[self.TARGET_COL]])

        return train, val, test

    # LSTM sequence generation

    # Note:
    # Sequence construction necessarily reduces the number of usable samples
    # by `sequence_length`. This affects only temporal models; baseline models
    # operate on the full tabular dataset. This behaviour is expected and
    # documented to ensure transparency in model comparison.

    def _make_sequences(
            self, df: pd.DataFrame, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert tabular data into sliding window sequences for LSTM models.

        LSTMs require sequential input data. This method creates overlapping
        windows of length `seq_len`, where each window is used to predict
        the subsequent target value (sequence-to-one forecasting).

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed and scaled DataFrame.
        seq_len : int
            Length of input sequences (number of time steps).

        Returns
        -------
        X : np.ndarray
            Input sequences of shape (n_samples, seq_len, n_features).
        y : np.ndarray
            Target values of shape (n_samples, 1).
            Each target corresponds to the time step immediately following
            its input sequence.

        Notes
        -----
        Sequence construction:
        - Sample i uses features from time steps [i : i+seq_len]
        - Sample i predicts target at time step [i+seq_len]

        This formulation aligns with standard sequence-to-one forecasting
        setups used in LSTM time series prediction (Hochreiter and
        Schmidhuber, 1997; Abduallah et al., 2022).

        Example:
        If seq_len=10 and we have 1000 samples:
        - First sequence: samples 0-9 → predict sample 10
        - Second sequence: samples 1-10 → predict sample 11
        - ...
        - Last sequence: samples 989-998 → predict sample 999
        - Total sequences: 990 (reduced by seq_len from original)

        The reduction in sample count is expected and affects only temporal
        models. Baseline models use the full tabular dataset without
        sequence construction.
        """
        X, y = [], []

        # Extract feature matrix and target vector
        features = df[self.FEATURE_COLS].values
        target = df[self.TARGET_COL].values

        # Create sliding windows
        # Stop at len(df) - seq_len to ensure we have a target for each sequence
        for i in range(len(df) - seq_len):
            # Input sequence: features from time i to i+seq_len (exclusive)
            X.append(features[i:i + seq_len])

            # Target: value at time i+seq_len (one step ahead)
            y.append(target[i + seq_len])

        # Convert lists to NumPy arrays
        # X shape: (n_samples, seq_len, n_features)
        # y shape: (n_samples, 1)
        return np.array(X), np.array(y).reshape(-1, 1)

    # Public pipeline

    def run(
            self,
            input_csv: str = "data/raw/omni2_combined.csv",
            output_dir: str = "data/processed",
    ) -> Dict[str, int]:
        """
        Execute the full preprocessing pipeline and persist outputs.

        This method orchestrates the complete preprocessing workflow:
        1. Load raw OMNI data
        2. Validate schema and data quality
        3. Handle missing values
        4. Remove physical outliers
        5. Compute derived features (SSI, severity classes, etc.)
        6. Split data chronologically
        7. Fit scalers and standardise
        8. Generate LSTM sequences
        9. Persist all outputs (CSV for baselines, NumPy for LSTM, scalers)

        Parameters
        ----------
        input_csv : str, optional
            Path to raw OMNI CSV file.
            Default is "data/raw/omni2_combined.csv".
        output_dir : str, optional
            Directory for saving processed outputs.
            Default is "data/processed".

        Returns
        -------
        dict
            Summary statistics about the processed dataset:
            - train_samples: Number of LSTM training sequences
            - val_samples: Number of LSTM validation sequences
            - test_samples: Number of LSTM test sequences
            - sequence_length: Length of LSTM input sequences
            - n_features: Number of input features
            - target: Name of target variable

        Outputs
        -------
        The following files are created in output_dir:

        Baseline model data (CSV format):
        - train_baseline.csv: Training set (scaled)
        - val_baseline.csv: Validation set (scaled)
        - test_baseline.csv: Test set (scaled)

        LSTM model data (NumPy format):
        - X_train.npy: Training sequences (n_samples, seq_len, n_features)
        - y_train.npy: Training targets (n_samples, 1)
        - X_val.npy: Validation sequences
        - y_val.npy: Validation targets
        - X_test.npy: Test sequences
        - y_test.npy: Test targets

        Scalers (for inverse transformation):
        - scaler_X.pkl: Fitted StandardScaler for features
        - scaler_y.pkl: Fitted StandardScaler for target

        Notes
        -----
        The pipeline enforces strict chronological ordering and prevents
        any form of data leakage. All transformations (filling, outlier
        removal, scaling) use only information available at each time step.

        Derived features are computed BEFORE scaling to maintain physical
        interpretability (Liemohn et al., 2021).
        """
        logger.info("Starting preprocessing pipeline")

        # Create output directory if it doesn't exist
        ensure_dir(output_dir)
        out = Path(output_dir)

        # Load raw OMNI data
        # parse_dates ensures datetime column is correctly typed
        df = pd.read_csv(input_csv, parse_dates=["datetime"])

        # Early validation ensures schema correctness before mutation
        # This catches issues like missing columns or wrong data types early
        validate_omni_dataframe(df)

        # Data cleaning
        df = self._handle_missing(df)  # Fill missing values
        df = self._remove_physical_outliers(df)  # Remove unphysical samples

        # 🔒 DERIVED FEATURES COMPUTED IN PHYSICAL SPACE
        # This is critical: we compute derived features (SSI, severity classes,
        # auroral latitude) BEFORE scaling to maintain physical interpretability.
        # Scaling is applied afterwards to the complete feature set.
        df = add_all_derived_features(df)

        # Chronological splitting (prevents temporal leakage)
        train, val, test = self._split(df)

        # Standardisation (fitted on training data only)
        train, val, test = self._scale(train, val, test)

        # Persist baseline (tabular) datasets
        # These are used by Linear Regression and Random Forest models
        train.to_csv(out / "train_baseline.csv", index=False)
        val.to_csv(out / "val_baseline.csv", index=False)
        test.to_csv(out / "test_baseline.csv", index=False)

        # Generate and persist LSTM-ready sequences
        # Extract sequence length from configuration
        seq_len = self.config["models"]["lstm"]["sequence_length"]

        # Create sequences for each split
        # Note: This reduces the number of samples by seq_len in each split
        X_train, y_train = self._make_sequences(train, seq_len)
        X_val, y_val = self._make_sequences(val, seq_len)
        X_test, y_test = self._make_sequences(test, seq_len)

        # Persist NumPy arrays for LSTM training
        # Binary format is more efficient than CSV for large arrays
        np.save(out / "X_train.npy", X_train)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "X_val.npy", X_val)
        np.save(out / "y_val.npy", y_val)
        np.save(out / "X_test.npy", X_test)
        np.save(out / "y_test.npy", y_test)

        # Persist scalers to enable inverse transformation of predictions
        # during evaluation and deployment.
        # Inverse transformation is essential for:
        # 1. Computing metrics in original units (interpretable RMSE)
        # 2. Comparing predictions with observed values
        # 3. Deployment to operational forecasting systems
        with open(out / "scaler_X.pkl", "wb") as f:
            pickle.dump(self.scaler_X, f)

        with open(out / "scaler_y.pkl", "wb") as f:
            pickle.dump(self.scaler_y, f)

        # Compile summary statistics
        summary = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "sequence_length": seq_len,
            "n_features": len(self.FEATURE_COLS),
            "target": self.TARGET_COL,
        }

        logger.info(f"Preprocessing complete: {summary}")
        return summary


def main():
    """
    Entry point for preprocessing script.

    This function runs the complete preprocessing pipeline with default
    parameters, suitable for standard workflow execution.

    Usage
    -----
    Run from command line:
        python preprocess.py

    The script will:
    1. Load raw OMNI data from data/raw/omni2_combined.csv
    2. Clean and validate the data
    3. Compute derived features
    4. Split chronologically into train/val/test
    5. Standardise features and target
    6. Generate LSTM sequences
    7. Save all outputs to data/processed/

    Output files are consumed by downstream training scripts:
    - baseline_models.py (uses CSV files)
    - lstm_trainer.py (uses NumPy arrays)
    """
    DataPreprocessor().run()


if __name__ == "__main__":
    main()
