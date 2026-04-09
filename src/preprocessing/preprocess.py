"""Preprocessing pipeline: clean, feature-engineer, scale, and split OMNI2 data for all models."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.validators import validate_omni_dataframe
from src.features.derived_features import add_all_derived_features, assign_storm_severity_class
from src.utils import load_config, ensure_dir, setup_logging

logger = setup_logging()

#: Solar wind input features used by all models.
#: dst is included because it is the dominant SSI driver and is genuinely
#: available at prediction time — it does not constitute data leakage.
FEATURE_COLS: list[str] = ["bt", "bz_gsm", "speed", "density", "dst"]

#: Regression target: continuous Storm Severity Index in [0, 1].
TARGET_COL: str = "storm_severity_index"

#: Physical metadata columns persisted before scaling.
PHYSICAL_META_COLS: list[str] = [
    "datetime", "auroral_latitude_deg", "storm_severity_class", "storm_severity_index"
]


class DataPreprocessor:
    """Unified preprocessing pipeline for baseline and temporal ML models."""

    # Re-exported so callers can reach the constant via the class without an extra import.
    FEATURE_COLS = FEATURE_COLS
    TARGET_COL = TARGET_COL

    def __init__(self, config_path: str = "config.yaml"):
        """Initialise the data preprocessor."""
        self.config = load_config(config_path)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using forward- and backward-fill."""
        df = df.sort_values("datetime").reset_index(drop=True)
        cols = self.FEATURE_COLS
        df[cols] = df[cols].ffill().bfill()
        return df.dropna(subset=cols)

    def _remove_physical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove samples outside physically plausible limits defined in config.yaml."""
        limits = self.config["physical_limits"]
        for col, (low, high) in limits.items():
            if col in df.columns:
                df = df[(df[col] >= low) & (df[col] <= high)]
        return df

    def _resample_6hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to 6-hourly averages and re-derive storm severity class."""
        # Drop categorical column — cannot be averaged numerically.
        # It is re-derived from the resampled SSI below.
        df = df.drop(columns=["storm_severity_class"], errors="ignore")

        df = (
            df.set_index("datetime")
            .resample("6h")
            .mean(numeric_only=True)
            .dropna(how="all")
            .reset_index()
        )

        df = assign_storm_severity_class(df)

        logger.info("Resampled to 6-hourly cadence: %d rows.", len(df))
        return df

    def _split(
            self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronologically split data into train, validation, and test sets."""
        test_frac = self.config["training"]["test_split"]
        val_frac = self.config["training"]["validation_split"]

        n = len(df)
        test_sample_count = int(n * test_frac)
        val_sample_count = int(n * val_frac)
        train_sample_count = n - test_sample_count - val_sample_count

        train = df.iloc[:train_sample_count].copy()
        val = df.iloc[train_sample_count:train_sample_count + val_sample_count].copy()
        test = df.iloc[train_sample_count + val_sample_count:].copy()

        return train, val, test

    def _scale(
            self,
            train: pd.DataFrame,
            val: pd.DataFrame,
            test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit scalers on training data only and apply to all splits."""
        # Scalers fitted on training data only to prevent leakage from
        # validation and test splits into the feature normalisation.
        self.scaler_X.fit(train[self.FEATURE_COLS])
        self.scaler_y.fit(train[[self.TARGET_COL]])

        for df in (train, val, test):
            df[self.FEATURE_COLS] = self.scaler_X.transform(df[self.FEATURE_COLS])
            df[self.TARGET_COL] = self.scaler_y.transform(df[[self.TARGET_COL]])

        return train, val, test

    def _make_sequences(
            self, df: pd.DataFrame, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert tabular data into sliding-window sequences for LSTM/GRU.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed and scaled DataFrame.
        seq_len : int
            Number of 6-hourly time steps per input window.

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

    def run(
            self,
            input_csv: str = "data/raw/omni2_combined.csv",
            output_dir: str = "data/processed",
    ) -> dict[str, object]:
        """Execute the full preprocessing pipeline and persist outputs."""
        logger.info("Starting preprocessing pipeline.")

        ensure_dir(output_dir)
        output_path = Path(output_dir)

        df = pd.read_csv(input_csv, parse_dates=["datetime"])
        validate_omni_dataframe(df)

        df = self._handle_missing(df)
        df = self._remove_physical_outliers(df)

        # SSI computed at hourly resolution before resampling — computing after
        # resampling would dilute extreme-event peaks by averaging them with
        # surrounding quieter hours, preventing the severe/extreme SSI bands
        # from being populated in training data.
        df = add_all_derived_features(df)
        logger.info(
            "SSI computed at hourly resolution. "
            "Max SSI before resampling: %.4f", df["storm_severity_index"].max()
        )

        if self.config.get("data", {}).get("resample_6h", True):
            df = self._resample_6hourly(df)
            logger.info(
                "Max SSI after resampling: %.4f", df["storm_severity_index"].max()
            )

        assert not df[self.FEATURE_COLS + [self.TARGET_COL]].isnull().any().any(), (
            "Missing values remain after imputation — check pipeline."
        )

        train, val, test = self._split(df)
        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train), len(val), len(test),
        )

        # Physical metadata saved before StandardScaler transform — auroral
        # latitude and storm class are derived from the true [0,1] SSI, not
        # the zero-mean scaled version.
        test_meta = test[[c for c in PHYSICAL_META_COLS if c in test.columns]].copy()
        test_meta.to_csv(output_path / "test_meta.csv", index=False)
        logger.info(
            "Physical test metadata saved → test_meta.csv "
            "(max SSI: %.4f)", test["storm_severity_index"].max()
        )

        train, val, test = self._scale(train, val, test)

        train.to_csv(output_path / "train_baseline.csv", index=False)
        val.to_csv(output_path / "val_baseline.csv", index=False)
        test.to_csv(output_path / "test_baseline.csv", index=False)
        logger.info("Baseline CSVs saved (scaled).")

        # Separate sequence arrays per model — each model reads its own
        # sequence_length from config so the arrays may differ in length.
        for model_key in ("lstm", "gru"):
            seq_len = self.config["models"][model_key]["sequence_length"]

            X_tr, y_tr = self._make_sequences(train, seq_len)
            X_vl, y_vl = self._make_sequences(val, seq_len)
            X_te, y_te = self._make_sequences(test, seq_len)

            np.save(output_path / f"X_train_{model_key}.npy", X_tr)
            np.save(output_path / f"y_train_{model_key}.npy", y_tr)
            np.save(output_path / f"X_val_{model_key}.npy", X_vl)
            np.save(output_path / f"y_val_{model_key}.npy", y_vl)
            np.save(output_path / f"X_test_{model_key}.npy", X_te)
            np.save(output_path / f"y_test_{model_key}.npy", y_te)

            logger.info(
                "%s sequences: train=%d  val=%d  test=%d  (seq_len=%d)",
                model_key.upper(), len(X_tr), len(X_vl), len(X_te), seq_len,
            )

        with open(output_path / "scaler_X.pkl", "wb") as file_handle:
            pickle.dump(self.scaler_X, file_handle)
        with open(output_path / "scaler_y.pkl", "wb") as file_handle:
            pickle.dump(self.scaler_y, file_handle)
        logger.info("Scalers saved.")

        lstm_seq = self.config["models"]["lstm"]["sequence_length"]
        gru_seq = self.config["models"]["gru"]["sequence_length"]

        summary = {
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "lstm_sequence_length": lstm_seq,
            "gru_sequence_length": gru_seq,
            "n_features": len(self.FEATURE_COLS),
            "target_name": self.TARGET_COL,
            "feature_names": self.FEATURE_COLS.copy(),
        }

        logger.info("Preprocessing complete: %s", summary)
        return summary


def main():
    """Run the preprocessing pipeline as a script entry point."""
    DataPreprocessor().run()


if __name__ == "__main__":
    main()
