"""Linear Regression and Random Forest baselines for Storm Severity Index prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing.preprocess import FEATURE_COLS, TARGET_COL
from src.utils import load_config, ensure_dir, setup_logging

logger = setup_logging()


class BaselineTrainer:
    """Train and evaluate non-temporal regression baselines on the Storm Severity Index."""

    # Re-exported so callers can reach these constants via the class without an extra import.
    FEATURE_COLS = FEATURE_COLS
    TARGET_COL = TARGET_COL

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        rf_cfg = self.config["models"]["baseline"]["random_forest"]

        self.models = {
            "linear_regression": LinearRegression(
                fit_intercept=self.config["models"]["baseline"]["linear_regression"].get(
                    "fit_intercept", True
                )
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=rf_cfg["n_estimators"],
                max_depth=rf_cfg["max_depth"],
                random_state=self.config["training"]["random_seed"],
                n_jobs=-1,
            ),
        }

    def _load_split(self, path: Path):
        """Load feature matrix and target vector from a preprocessed CSV split."""
        df = pd.read_csv(path)
        X = df[self.FEATURE_COLS].values
        y = df[self.TARGET_COL].values
        return X, y

    @staticmethod
    def _evaluate(y_true, y_pred) -> dict[str, float]:
        """Compute RMSE, MAE, and R² for a set of predictions."""
        mse = mean_squared_error(y_true, y_pred)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def run(
            self,
            processed_dir: str = "data/processed",
            output_dir: str = "outputs/baselines",
    ) -> dict[str, dict[str, float]]:
        """Train both baselines, evaluate on the test set, and serialise artefacts.

        Parameters
        ----------
        processed_dir
            Directory containing preprocessed CSV files.
        output_dir
            Directory for saving trained models and prediction CSVs.

        Returns
        -------
        dict
            Nested mapping of model name to evaluation metrics
            (``rmse``, ``mae``, ``r2``).
        """
        processed = Path(processed_dir)
        out = Path(output_dir)

        ensure_dir(out)
        ensure_dir(out / "models")
        ensure_dir(out / "predictions")

        logger.info("Loading preprocessed baseline datasets.")

        X_train, y_train = self._load_split(processed / "train_baseline.csv")
        X_test, y_test = self._load_split(processed / "test_baseline.csv")

        with open(processed / "scaler_y.pkl", "rb") as fh:
            scaler_y = pickle.load(fh)

        results: dict[str, dict[str, float]] = {}

        for name, model in self.models.items():
            logger.info("Training baseline model: %s", name)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = self._evaluate(y_test, y_pred)
            results[name] = metrics

            with open(out / "models" / f"{name}.pkl", "wb") as fh:
                pickle.dump(model, fh)

            y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            pred_df = pd.DataFrame({
                "model": name,
                "y_true": y_test_inv,
                "y_pred": y_pred_inv,
            })
            pred_df.to_csv(
                out / "predictions" / f"{name}_test_predictions.csv",
                index=False,
            )

            logger.info("%s metrics: %s", name, metrics)

        return results


def main():
    """Train baseline models, print evaluation metrics, and save artefacts."""
    trainer = BaselineTrainer()
    results = trainer.run()

    print("\nBaseline model evaluation (SSI):")
    for model, metrics in results.items():
        print(f"\n{model}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
