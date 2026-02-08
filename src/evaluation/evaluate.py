"""
Model evaluation and visualisation utilities.

This module evaluates trained models by comparing their predictions
against ground-truth targets. It computes standard regression metrics
and produces publication-quality plots for analysis and reporting.

Current scope:
- Baseline SSI regression models (Linear Regression, Random Forest)

Future extensions:
- Sequence models (LSTM, GRU)
- Multi-horizon forecasts
- Model comparison across feature sets

Design principles:
- No data leakage
- Read-only access to processed data
- Deterministic, reproducible evaluation

References:
- Liemohn et al. (2021) - RMSE and robust evaluation metrics for space physics
- Pedregosa et al. (2011) - scikit-learn metrics implementation
- Hunter (2007) - Matplotlib visualisation library
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.persistence import persistence_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Metric computation

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics for model evaluation.

    This function calculates three complementary metrics that together
    provide a comprehensive assessment of model performance:

    1. RMSE - Penalises large errors, important for storm forecasting
    2. MAE - Robust to outliers, interpretable in original units
    3. R² - Proportion of variance explained, model fit quality

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values (actual SSI observations).
    y_pred : np.ndarray
        Model predictions (predicted SSI values).

    Returns
    -------
    dict
        Dictionary containing:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r2: Coefficient of determination (R²)

    Notes
    -----
    RMSE is the primary metric as recommended by Liemohn et al. (2021)
    for magnetospheric physics model evaluation. It emphasises the
    importance of accurately predicting extreme events (large errors
    are penalised quadratically).

    MAE provides a complementary view with linear error penalisation,
    making it more robust to outliers and easier to interpret (average
    absolute deviation from truth).

    R² indicates how well the model captures variance in the target
    variable. Values range from -∞ to 1, where:
    - 1.0 = perfect predictions
    - 0.0 = model performs no better than predicting the mean
    - Negative = model performs worse than mean baseline

    All metrics are computed using scikit-learn's implementations
    (Pedregosa et al., 2011).
    """
    return {
        # Root Mean Squared Error: √(mean((y_true - y_pred)²))
        # Primary metric for geomagnetic forecasting (Liemohn et al., 2021)
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),

        # Mean Absolute Error: mean(|y_true - y_pred|)
        # Robust to outliers, interpretable in SSI units
        "mae": mean_absolute_error(y_true, y_pred),

        # Coefficient of determination: 1 - (SS_res / SS_tot)
        # Proportion of variance explained by the model
        "r2": r2_score(y_true, y_pred),
    }


# Plotting utilities


def plot_timeseries(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
        n_points: int = 500,
):
    """
    Plot predicted vs true SSI as a time series.

    This visualisation shows how well the model tracks the temporal
    evolution of storm severity. It enables visual assessment of:
    - Overall prediction accuracy
    - Temporal lag or lead in predictions
    - Systematic over/under-prediction patterns
    - Ability to capture storm onset and recovery

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth SSI values (actual observations).
    y_pred : np.ndarray
        Predicted SSI values (model outputs).
    model_name : str
        Name of the evaluated model (e.g., "linear_regression").
    output_path : Path
        File path to save the figure (PNG format).
    n_points : int, optional
        Number of points to plot for readability.
        Default is 500 (reduces visual clutter for long series).

    Notes
    -----
    The plot displays:
    - Blue line: Ground truth SSI (what actually happened)
    - Orange line: Model predictions (what the model forecasted)

    Good models show close agreement between the two lines throughout
    the time series. Systematic deviations indicate model bias or
    inability to capture certain storm characteristics.

    Only the first n_points samples are plotted to maintain readability.
    For complete analysis, examine the full prediction CSV files.

    Figure is saved using Matplotlib (Hunter, 2007) with tight layout
    to minimise whitespace in publications.
    """
    # Create figure with publication-appropriate dimensions
    # 12×4 inches provides good aspect ratio for time series
    plt.figure(figsize=(12, 4))

    # Plot ground truth
    # linewidth=2 ensures visibility in print
    plt.plot(y_true[:n_points], label="True SSI", linewidth=2)

    # Plot predictions
    # alpha=0.8 provides slight transparency to see overlaps
    plt.plot(y_pred[:n_points], label="Predicted SSI", alpha=0.8)

    # Label axes with physical interpretation
    plt.xlabel("Time step")
    plt.ylabel("Storm Severity Index (SSI)")

    # Title identifies the model being evaluated
    plt.title(f"{model_name}: SSI prediction vs truth")

    # Legend distinguishes true from predicted
    plt.legend()

    # Tight layout removes excess whitespace
    plt.tight_layout()

    # Save to disk (PNG format for publications)
    plt.savefig(output_path)

    # Close figure to free memory
    plt.close()


def plot_scatter(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
):
    """
    Plot predicted vs true SSI as a scatter plot.

    Scatter plots provide a complementary view to time series by showing
    the relationship between predictions and observations without temporal
    structure. This enables assessment of:
    - Overall prediction accuracy (proximity to diagonal)
    - Systematic bias (consistent over/under-prediction)
    - Heteroscedasticity (error variance changes with SSI magnitude)
    - Outliers (points far from diagonal)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth SSI values (x-axis).
    y_pred : np.ndarray
        Predicted SSI values (y-axis).
    model_name : str
        Name of the evaluated model.
    output_path : Path
        File path to save the figure (PNG format).

    Notes
    -----
    The plot includes:
    - Scatter points: Each point represents one prediction
    - Diagonal line: Perfect prediction line (y=x)
    - Points on the line = perfect predictions
    - Points above the line = over-predictions
    - Points below the line = under-predictions

    Alpha transparency (0.4) reveals density in regions with many points.
    Clustering tightly around the diagonal indicates good performance.

    For geomagnetic forecasting, it's particularly important to examine
    behaviour at high SSI values (severe storms), as errors in this
    regime have greater operational consequences (Liemohn et al., 2021).

    Square aspect ratio (5×5 inches) ensures the diagonal appears at 45°,
    facilitating visual assessment of prediction bias.
    """
    # Create square figure for unbiased visual assessment
    plt.figure(figsize=(5, 5))

    # Scatter plot with transparency to show density
    # alpha=0.4 reveals overlapping points
    plt.scatter(y_true, y_pred, alpha=0.4)

    # Add perfect prediction line (y=x)
    # This diagonal serves as a reference: points should cluster here
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color="red")

    # Label axes clearly
    plt.xlabel("True SSI")
    plt.ylabel("Predicted SSI")

    # Title identifies the model
    plt.title(f"{model_name}: Predicted vs true SSI")

    # Tight layout for publication quality
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)

    # Close to free memory
    plt.close()


# Main evaluation routine


def evaluate_baseline_models(
        processed_dir: Path,
        results_dir: Path,
):
    """
    Evaluate baseline SSI regression models.

    This function orchestrates the complete evaluation workflow for
    baseline models (Linear Regression, Random Forest, Persistence):

    1. Discovers prediction files from trained models
    2. Loads predictions and ground truth
    3. Computes regression metrics (RMSE, MAE, R²)
    4. Generates diagnostic plots (time series, scatter)
    5. Evaluates persistence (t → t+1) baseline
    6. Saves metrics summary and figures to disk

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed datasets.
        Used for loading test targets for the persistence baseline.
    results_dir : Path
        Directory containing model predictions and outputs.

    Outputs
    -------
    Files:
    - metrics_baselines.csv: Summary table of all model metrics

    Plots (in outputs/baselines/plots/):
    - {model_name}_timeseries.png
    - {model_name}_scatter.png

    Notes
    -----
    Persistence is evaluated as a naïve temporal lower bound:
        ŷ[t+1] = y[t]

    It is not trained, serialised, or tuned, and is evaluated under
    identical conditions to learned baselines.
    """

    # Discover baseline prediction files

    candidate_dirs = [
        results_dir / "predictions",  # Used in unit tests
        results_dir / "baselines" / "predictions",  # Used in production runs
    ]

    prediction_files = []
    for d in candidate_dirs:
        if d.exists():
            prediction_files.extend(d.glob("*_test_predictions.csv"))

    if not prediction_files:
        raise RuntimeError(
            "No baseline prediction files found. "
            "Expected '*_test_predictions.csv' under outputs/predictions "
            "or outputs/baselines/predictions."
        )

    # Create output directories

    plots_dir = results_dir / "baselines" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = results_dir / "baselines" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # List to accumulate metrics from all models
    metrics_rows = []

    # Evaluate learned baseline models (Linear Regression, Random Forest)

    for pred_file in prediction_files:
        # Extract model name from filename
        # E.g., "linear_regression_test_predictions.csv" → "linear_regression"
        model_name = pred_file.stem.replace("_test_predictions", "")

        # Load predictions and ground truth
        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        # Compute regression metrics
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = model_name
        metrics_rows.append(metrics)

        # Generate diagnostic plots
        plot_timeseries(
            y_true,
            y_pred,
            model_name,
            plots_dir / f"{model_name}_timeseries.png",
        )

        plot_scatter(
            y_true,
            y_pred,
            model_name,
            plots_dir / f"{model_name}_scatter.png",
        )

    # Evaluate persistence (t → t+1) baseline

    test_csv = processed_dir / "test_baseline.csv"

    if test_csv.exists():
        # Load test target series
        df_test = pd.read_csv(test_csv)
        y = df_test["storm_severity_index"].values

        # Generate persistence predictions
        y_true, y_pred = persistence_forecast(y)

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = "persistence"
        metrics_rows.append(metrics)

        # Save persistence predictions
        pd.DataFrame({
            "model": "persistence",
            "y_true": y_true,
            "y_pred": y_pred,
        }).to_csv(
            preds_dir / "persistence_test_predictions.csv",
            index=False,
        )

        # Generate persistence plots
        plot_timeseries(
            y_true,
            y_pred,
            "persistence",
            plots_dir / "persistence_timeseries.png",
        )

        plot_scatter(
            y_true,
            y_pred,
            "persistence",
            plots_dir / "persistence_scatter.png",
        )

    # Save consolidated metrics table

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .set_index("model")
        .sort_index()
    )

    metrics_df.to_csv(results_dir / "metrics_baselines.csv")


if __name__ == "__main__":
    """
    Command-line interface for baseline model evaluation.

    This block enables the script to be run directly from the command line:
        python evaluate.py

    It automatically determines the project root directory and evaluates
    all baseline models found in the outputs directory.

    Usage
    -----
    From project root:
        python src/evaluate.py

    Or from src directory:
        python evaluate.py

    The script will:
    1. Discover all baseline prediction files
    2. Compute metrics for each model
    3. Generate diagnostic plots
    4. Save metrics_baselines.csv

    Prerequisites
    -------------
    - Baseline models must have been trained (baseline_models.py)
    - Prediction files must exist in outputs/baselines/predictions/

    Outputs
    -------
    - outputs/metrics_baselines.csv: Metrics summary table
    - outputs/baselines/plots/*.png: Diagnostic visualisations
    """
    # Determine project root (two levels up from this file)
    # This works regardless of where the script is run from
    project_root = Path(__file__).resolve().parents[1]

    # Run evaluation with standard directory structure
    evaluate_baseline_models(
        processed_dir=PROJECT_ROOT / "data" / "processed",
        results_dir=PROJECT_ROOT / "outputs",
    )
