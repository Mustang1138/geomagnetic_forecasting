"""
Baseline regression models for geomagnetic storm severity forecasting.

This module implements conservative, non-temporal machine learning
baselines trained to predict the Storm Severity Index (SSI).

Design principles:
- No additional preprocessing beyond frozen pipeline outputs
- Fixed, interpretable model configurations
- Regression-first (continuous severity prediction)
- Fully reproducible and comparable with temporal models

References:
- Breiman (2001) - Random Forest methodology
- Pedregosa et al. (2011) - scikit-learn implementation
- Liemohn et al. (2021) - Evaluation metrics for magnetospheric physics
"""

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import load_config, ensure_dir, setup_logging

logger = setup_logging()


class BaselineTrainer:
    """
    Train and evaluate baseline regression models on SSI.

    This class implements two baseline models:
    1. Linear Regression - provides a simple, interpretable baseline
    2. Random Forest - captures non-linear relationships whilst remaining interpretable

    The baseline models serve as benchmarks against which more complex temporal
    models (e.g., LSTMs) can be compared, following best practices for time series
    model evaluation (Cerqueira et al., 2020).
    """

    # Feature columns used for prediction
    # These correspond to key solar wind parameters that drive geomagnetic activity
    # (Burton et al., 1975; Gonzalez et al., 1994)
    FEATURE_COLS = ["bz_gsm", "speed", "density"]

    # Target variable: continuous Storm Severity Index in [0, 1]
    TARGET_COL = "storm_severity_index"

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise the baseline trainer with model configurations.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file containing hyperparameters.
            Default is "config.yaml".

        Notes
        -----
        Model hyperparameters are intentionally fixed (not tuned) to ensure
        reproducibility and fair comparison. This follows recommendations from
        Liemohn et al. (2021) for robust magnetospheric physics model comparisons.
        """
        self.config = load_config(config_path)

        # Extract Random Forest configuration from config file
        rf_cfg = self.config["models"]["baseline"]["random_forest"]

        # Initialise models with fixed hyperparameters
        self.models = {
            # Linear Regression: ordinary least squares regression
            # Provides an interpretable baseline and tests the linear hypothesis
            # (Pedregosa et al., 2011)
            "linear_regression": LinearRegression(
                fit_intercept=self.config["models"]["baseline"]["linear_regression"].get(
                    "fit_intercept",
                    True)),

            # Random Forest: ensemble of decision trees
            # Captures non-linear relationships whilst maintaining interpretability
            # through feature importance (Breiman, 2001)
            "random_forest": RandomForestRegressor(
                n_estimators=rf_cfg["n_estimators"],  # Number of trees in the forest
                max_depth=rf_cfg["max_depth"],  # Maximum tree depth (prevents overfitting)
                random_state=self.config["training"]["random_seed"],  # Ensures reproducibility
                n_jobs=-1,  # Use all available CPU cores for parallel training
            ),
        }

    # Data loading

    def _load_split(self, path: Path):
        """
        Load a preprocessed data split from CSV.

        Parameters
        ----------
        path : Path
            Path to the CSV file containing preprocessed data.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).

        Notes
        -----
        This method expects data that has already been cleaned, scaled, and
        split by the preprocessing pipeline to prevent any data leakage
        (Cerqueira et al., 2020).
        """
        df = pd.read_csv(path)

        # Extract feature matrix
        # Features are already scaled by the preprocessing pipeline
        X = df[self.FEATURE_COLS].values

        # Extract target vector
        y = df[self.TARGET_COL].values

        return X, y

    # Evaluation

    @staticmethod
    def _evaluate(y_true, y_pred) -> Dict[str, float]:
        """
        Compute regression evaluation metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth target values.
        y_pred : array-like
            Predicted target values.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: Coefficient of determination

        Notes
        -----
        RMSE is the primary metric as recommended by Liemohn et al. (2021)
        for magnetospheric physics model evaluation. MAE provides additional
        insight into average prediction error, whilst R² indicates the proportion
        of variance explained by the model.
        """
        # Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)

        # Root Mean Squared Error (primary metric)
        # RMSE penalises large errors more heavily than MAE, which is important
        # for geomagnetic storm forecasting where extreme events matter most
        # (Liemohn et al., 2021)
        rmse = float(np.sqrt(mse))

        return {
            "rmse": rmse,  # Root Mean Squared Error
            "mae": mean_absolute_error(y_true, y_pred),  # Mean Absolute Error
            "r2": r2_score(y_true, y_pred),  # R-squared (coefficient of determination)
        }

    def run(
            self,
            processed_dir: str = "data/processed",
            output_dir: str = "outputs/baselines",
    ) -> Dict[str, Dict[str, float]]:
        """
        Execute the complete baseline model training and evaluation pipeline.

        This method:
        1. Loads preprocessed training and test data
        2. Trains each baseline model
        3. Generates predictions on the test set
        4. Computes evaluation metrics
        5. Persists trained models and predictions

        Parameters
        ----------
        processed_dir : str, optional
            Directory containing preprocessed CSV files.
            Default is "data/processed".
        output_dir : str, optional
            Directory for saving trained models, predictions, and outputs.
            Default is "outputs/baselines".

        Returns
        -------
        dict
            Nested dictionary mapping model names to their evaluation metrics.
            Example: {"linear_regression": {"rmse": 0.123, "mae": 0.098, "r2": 0.85}}

        Notes
        -----
        The validation split is intentionally not used for baseline models.
        Baselines are trained once with fixed hyperparameters and evaluated
        only on the held-out test set. This differs from temporal models
        which may use the validation set for early stopping or hyperparameter
        tuning (Hochreiter and Schmidhuber, 1997).
        """
        # Convert string paths to Path objects for robust path handling
        processed = Path(processed_dir)
        out = Path(output_dir)

        # Create output directory structure
        ensure_dir(out)
        ensure_dir(out / "models")  # For serialised model objects
        ensure_dir(out / "predictions")  # For prediction CSV files

        logger.info("Loading preprocessed baseline datasets")

        # Load training data
        # This data has already been cleaned, scaled, and chronologically split
        # by the preprocessing pipeline (preprocess.py)
        X_train, y_train = self._load_split(processed / "train_baseline.csv")

        # Load test data
        # The test set represents future unseen data for final model evaluation
        # Note: Validation split is intentionally not used here.
        # Baselines are trained once with fixed hyperparameters
        # and evaluated only on the held-out test set.
        X_test, y_test = self._load_split(processed / "test_baseline.csv")

        # Dictionary to store evaluation outputs
        results = {}

        # Train and evaluate each baseline model
        for name, model in self.models.items():
            logger.info(f"Training baseline model: {name}")

            # Fit the model on training data
            # For Linear Regression: solves ordinary least squares
            # For Random Forest: builds ensemble of decision trees (Breiman, 2001)
            model.fit(X_train, y_train)

            # Generate predictions on test set
            y_pred = model.predict(X_test)

            # Compute evaluation metrics
            metrics = self._evaluate(y_test, y_pred)
            results[name] = metrics

            # Persist trained model for later use or deployment
            # Models are serialised using pickle for Python compatibility
            # (Pedregosa et al., 2011)
            with open(out / "models" / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)

            # Persist test set predictions for detailed error analysis
            # Storing predictions alongside ground truth enables:
            # - Residual analysis
            # - Error distribution visualisation
            # - Comparison with other models
            #
            # The 'model' column is included to support potential future
            # consolidation of predictions from multiple models into a single
            # DataFrame for comparative analysis (not currently used by
            # evaluate.py but included for forward compatibility).
            pred_df = pd.DataFrame({
                "model": name,  # Model identifier (for potential future use)
                "y_true": y_test,  # Ground truth SSI values
                "y_pred": y_pred,  # Model predictions
            })
            pred_df.to_csv(
                out / "predictions" / f"{name}_test_predictions.csv",
                index=False,
            )

            logger.info(f"{name} outputs: {metrics}")

        return results


def main():
    """
    Entry point for baseline model training script.

    This function instantiates the BaselineTrainer, executes the training
    pipeline, and prints formatted outputs to the console.

    Usage
    -----
    Run from command line:
        python baseline_models.py

    The script will:
    1. Load configuration from config.yaml
    2. Train Linear Regression and Random Forest models
    3. Evaluate on test set
    4. Save models and predictions
    5. Print evaluation metrics
    """
    # Instantiate trainer with default configuration
    trainer = BaselineTrainer()

    # Run complete training and evaluation pipeline
    results = trainer.run()

    # Print formatted outputs to console
    print("\nBaseline model evaluation (SSI):")
    for model, metrics in results.items():
        print(f"\n{model}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
