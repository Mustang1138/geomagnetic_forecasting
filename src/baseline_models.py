"""
Baseline machine learning models for geomagnetic forecasting.

Implements classical, non-temporal models to establish reference
performance against which more complex temporal models (e.g. LSTM)
are compared.

Models implemented:
- Linear Regression
- Random Forest Regressor

Design principles:
- Consume frozen, preprocessed baseline datasets
- No additional data cleaning or scaling
- Deterministic configuration via config.yaml
- Predictions saved in physical units (nT)

References:
- Breiman (2001)
- Pedregosa et al. (2011)
- Liemohn et al. (2021)
"""

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.utils import load_config, ensure_dir, setup_logging

logger = setup_logging()


class BaselineTrainer:
    """
    Train and evaluate baseline regression models using
    preprocessed tabular datasets.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        # Feature/target definitions are fixed to match preprocessing output
        # and established geomagnetic forecasting literature
        # (Abduallah et al., 2022; Zou et al., 2024).
        self.feature_cols = ["bz_gsm", "speed", "density"]
        self.target_col = "dst"

        self.output_dir = Path("results/baselines")
        ensure_dir(self.output_dir)

    # Data loading

    def _load_split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a preprocessed baseline split from disk.

        Parameters:
        name : str
            One of {'train', 'val', 'test'}.

        Returns:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector (2D, shape: [n_samples, 1]).
        """
        processed_dir = Path(self.config["data"]["processed_dir"])
        path = processed_dir / f"{name}_baseline.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing baseline dataset: {path}")

        df = pd.read_csv(path)

        missing = set(self.feature_cols + [self.target_col]) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        X = df[self.feature_cols].values
        y = df[self.target_col].values.reshape(-1, 1)

        return X, y

    @staticmethod
    def _fit_and_predict(
            model,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            X_test: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Fit a regression model and generate predictions for all data splits.

        Targets are flattened for compatibility with scikit-learn estimators,
        and predictions are reshaped to (n_samples, 1) to enforce a consistent
        output interface across baseline models.
        """
        model.fit(X_train, y_train.ravel())

        return {
            "train": model.predict(X_train).reshape(-1, 1),
            "val": model.predict(X_val).reshape(-1, 1),
            "test": model.predict(X_test).reshape(-1, 1),
        }

    # Models

    def train_linear_regression(self) -> Dict[str, np.ndarray]:
        """
        Train ordinary least squares linear regression.

        Used as a simple, interpretable baseline
        (Pedregosa et al., 2011).
        """
        logger.info("Training Linear Regression")

        X_train, y_train = self._load_split("train")

        # Validation data is loaded for pipeline consistency and
        # comparability with later temporal models. No hyperparameter
        # tuning is performed for baseline models by design.
        X_val, y_val = self._load_split("val")

        X_test, y_test = self._load_split("test")

        model = LinearRegression()
        preds = self._fit_and_predict(
            model,
            X_train,
            y_train,
            X_val,
            X_test,
        )

        self._save_outputs("linear_regression", model, preds)
        return preds

    def train_random_forest(self) -> Dict[str, np.ndarray]:
        """
        Train Random Forest regressor using fixed hyperparameters.

        The model is intentionally not exhaustively tuned, as it serves
        as a classical ML baseline rather than an optimised predictor
        (Breiman, 2001).
        """
        cfg = self.config["models"]["baseline"]["random_forest"]

        logger.info("Training Random Forest")

        X_train, y_train = self._load_split("train")

        # Validation data is loaded for consistency and future extensibility;
        # no hyperparameter tuning is performed for baseline models.
        X_val, y_val = self._load_split("val")

        X_test, y_test = self._load_split("test")

        model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )

        preds = self._fit_and_predict(
            model,
            X_train,
            y_train,
            X_val,
            X_test,
        )

        self._save_outputs("random_forest", model, preds)
        return preds

    # Persistence

    def _save_outputs(
            self,
            name: str,
            model,
            preds: Dict[str, np.ndarray],
    ) -> None:
        model_dir = self.output_dir / name
        ensure_dir(model_dir)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        for split, arr in preds.items():
            np.save(model_dir / f"y_pred_{split}.npy", arr)

        logger.info(f"Saved outputs for {name}")


if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.train_linear_regression()
    trainer.train_random_forest()
