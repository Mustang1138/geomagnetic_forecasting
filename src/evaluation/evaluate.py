"""Evaluation orchestrator: loads prediction CSVs, computes metrics, and generates plots."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.plots import (
    MODEL_COLOURS,
    plot_feature_importance,
    plot_model_ranking,
    plot_residual_distribution,
    plot_residuals,
    plot_scatter,
    plot_timeseries,
)
from src.models.persistence import persistence_forecast
from src.utils import load_pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R² in physical SSI units."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_baseline_models(processed_dir: Path, results_dir: Path) -> None:
    """Evaluate all baseline prediction CSVs and write per-model plots and a metrics CSV."""
    candidate_dirs = [
        results_dir / "predictions",
        results_dir / "baselines" / "predictions",
    ]

    prediction_files = []
    for d in candidate_dirs:
        if d.exists():
            prediction_files.extend(d.glob("*_test_predictions.csv"))

    if not prediction_files:
        raise RuntimeError("No baseline prediction files found.")

    plots_dir = results_dir / "baselines" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("_test_predictions", "")

        if model_name == "persistence":
            continue

        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = model_name
        metrics_rows.append(metrics)

        colour = MODEL_COLOURS.get(model_name)
        plot_timeseries(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_timeseries.png", colour=colour,
        )
        plot_scatter(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_scatter.png", colour=colour,
        )
        plot_residuals(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_residuals.png", colour=colour,
        )
        plot_residual_distribution(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_residual_hist.png", colour=colour,
        )

        if model_name == "random_forest":
            model_path = results_dir / "baselines" / "models" / "random_forest.pkl"

            if model_path.exists():
                model = load_pickle(model_path)
                train_csv = processed_dir / "train_baseline.csv"

                if train_csv.exists():
                    df_train = pd.read_csv(train_csv)
                    feature_cols = [
                        col for col in df_train.columns
                        if col not in {"datetime", "storm_severity_index",
                                       "storm_severity_class", "auroral_latitude_deg"}
                    ]
                    plot_feature_importance(model, feature_cols,
                                            plots_dir / "random_forest_feature_importance.png")

    test_csv = processed_dir / "test_baseline.csv"

    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        scaler_y = load_pickle(processed_dir / "scaler_y.pkl")

        y_scaled = df_test["storm_severity_index"].values
        y_true_scaled, y_pred_scaled = persistence_forecast(y_scaled)

        y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = "persistence"
        metrics_rows.append(metrics)

        colour = MODEL_COLOURS.get("persistence")
        plot_timeseries(
            y_true, y_pred, "persistence",
            plots_dir / "persistence_timeseries.png", colour=colour,
        )
        plot_scatter(
            y_true, y_pred, "persistence",
            plots_dir / "persistence_scatter.png", colour=colour,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    # Ensure 'model' is the first column so that callers reading the CSV with
    # index_col=0 get the model name as the index rather than a metric column.
    cols = ["model", "rmse", "mae", "r2"]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    metrics_df.to_csv(results_dir / "metrics_baselines.csv", index=False)


def evaluate_temporal_models(results_dir: Path) -> list[dict]:
    """Evaluate temporal model prediction CSVs and return a list of metric dicts."""
    temporal_pred_dir = results_dir / "temporal" / "predictions"

    if not temporal_pred_dir.exists():
        return []

    plots_dir = results_dir / "temporal" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    for pred_file in temporal_pred_dir.glob("*_predictions.csv"):
        model_name = pred_file.stem.replace("_predictions", "")

        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = model_name
        metrics_rows.append(metrics)

        colour = MODEL_COLOURS.get(model_name)
        plot_timeseries(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_timeseries.png", colour=colour,
        )
        plot_scatter(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_scatter.png", colour=colour,
        )
        plot_residuals(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_residuals.png", colour=colour,
        )
        plot_residual_distribution(
            y_true, y_pred, model_name,
            plots_dir / f"{model_name}_residual_hist.png", colour=colour,
        )

    return metrics_rows


def evaluate_all_models(processed_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Evaluate all models, write a combined metrics CSV, and return the results DataFrame."""
    evaluate_baseline_models(processed_dir, results_dir)
    baseline_df = pd.read_csv(results_dir / "metrics_baselines.csv")

    temporal_rows = evaluate_temporal_models(results_dir)
    temporal_df = pd.DataFrame(temporal_rows) if temporal_rows else pd.DataFrame()

    all_metrics = pd.concat([baseline_df, temporal_df], ignore_index=True)
    all_metrics = all_metrics.sort_values("rmse")
    all_metrics.to_csv(results_dir / "metrics_all_models.csv", index=False)

    plot_model_ranking(all_metrics, results_dir / "model_ranking_rmse.png")

    print("\nFinal Model Comparison:")
    print(all_metrics)

    return all_metrics


if __name__ == "__main__":
    evaluate_all_models(
        processed_dir=PROJECT_ROOT / "data" / "processed",
        results_dir=PROJECT_ROOT / "outputs",
    )
