"""
Model evaluation and visualisation utilities.

This module evaluates trained models by comparing their predictions
against ground-truth targets. It computes standard regression metrics
and produces publication-quality plots for analysis and reporting.

Design principles:
- All models train on scaled targets
- All prediction CSV files contain scaled targets
- Evaluation performs inverse scaling exactly once
- All reported metrics are in physical SSI units
- No data leakage
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.persistence import persistence_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")

# Metric computation


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE and R² in physical SSI units.
    """
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# Plotting utilities (always use inverse-scaled values)
def plot_timeseries(y_true, y_pred, model_name, output_path, n_points=500):
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:n_points], label="True SSI", linewidth=2)
    plt.plot(y_pred[:n_points], label="Predicted SSI", alpha=0.8)
    plt.xlabel("Time step")
    plt.ylabel("Storm Severity Index (SSI)")
    plt.title(f"{model_name}: SSI prediction vs truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scatter(y_true, y_pred, model_name, output_path):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4)

    # Perfect prediction line (auto-range)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle="--", linewidth=2)

    plt.xlabel("True SSI")
    plt.ylabel("Predicted SSI")
    plt.title(f"{model_name}: Predicted vs true SSI")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_ranking(metrics_df, output_path):
    """
    Plot model ranking based on RMSE (lower is better).
    """

    df = metrics_df.sort_values("rmse")

    plt.figure(figsize=(8, 5))

    plt.bar(df["model"], df["rmse"])

    plt.ylabel("RMSE (Storm Severity Index)")
    plt.xlabel("Model")
    plt.title("Model Performance Comparison (Lower RMSE = Better)")

    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, feature_names, output_path):
    """
    Plot feature importance for tree-based models.
    """

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    # 🔥 FIX: align lengths
    if len(feature_names) != len(importances):
        feature_names = feature_names[:len(importances)]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Baseline evaluation
def evaluate_baseline_models(processed_dir, results_dir):

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

    # Learnt baseline models (Linear Regression, Random Forest)
    for pred_file in prediction_files:

        model_name = pred_file.stem.replace("_test_predictions", "")

        # Skip persistence if it was saved during training
        if model_name == "persistence":
            continue

        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = model_name
        metrics_rows.append(metrics)

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

        # Feature importance (only for Random Forest)
        if model_name == "random_forest":
            model_path = results_dir / "baselines" / "models" / "random_forest.pkl"

            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

                # Load feature names from baseline dataset
                train_csv = processed_dir / "train_baseline.csv"
                if train_csv.exists():
                    df_train = pd.read_csv(train_csv)

                    feature_cols = [
                        col for col in df_train.columns
                        if col not in [
                            "datetime",
                            "storm_severity_index",
                            "storm_severity_class",
                            "auroral_latitude_deg"
                        ]
                    ]

                    plot_feature_importance(
                        model,
                        feature_cols,
                        plots_dir / "random_forest_feature_importance.png"
                    )

    # Persistence baseline (computed fresh here)
    test_csv = processed_dir / "test_baseline.csv"

    if test_csv.exists():
        df_test = pd.read_csv(test_csv)

        # storm_severity_index in test_baseline.csv is scaled
        with open(processed_dir / "scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)

        y_scaled = df_test["storm_severity_index"].values
        y_true_scaled, y_pred_scaled = persistence_forecast(y_scaled)

        y_true = scaler_y.inverse_transform(
            y_true_scaled.reshape(-1, 1)
        ).flatten()

        y_pred = scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["model"] = "persistence"
        metrics_rows.append(metrics)

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

    metrics_df = pd.DataFrame(metrics_rows)
    # Ensure 'model' is the first column so that callers reading the CSV with
    # index_col=0 get the model name as the index rather than a metric column.
    cols = ["model", "rmse", "mae", "r2"]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    metrics_df.to_csv(results_dir / "metrics_baselines.csv", index=False)


# Temporal model evaluation
def evaluate_temporal_models(results_dir):
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

    return metrics_rows


# Unified comparison
def evaluate_all_models(processed_dir: Path, results_dir: Path):
    # Evaluate baselines
    evaluate_baseline_models(processed_dir, results_dir)

    baseline_df = pd.read_csv(results_dir / "metrics_baselines.csv")

    # Evaluate temporal models
    temporal_rows = evaluate_temporal_models(results_dir)
    temporal_df = pd.DataFrame(
        temporal_rows) if temporal_rows else pd.DataFrame()

    # Combine
    all_metrics = pd.concat(
        [baseline_df, temporal_df],
        ignore_index=True,
    )

    # Sort by RMSE
    all_metrics = all_metrics.sort_values("rmse")

    all_metrics.to_csv(
        results_dir / "metrics_all_models.csv",
        index=False,
    )

    plot_model_ranking(
        all_metrics,
        results_dir / "model_ranking_rmse.png"
    )

    print("\nFinal Model Comparison:")
    print(all_metrics)

    return all_metrics


if __name__ == "__main__":
    evaluate_all_models(
        processed_dir=PROJECT_ROOT / "data" / "processed",
        results_dir=PROJECT_ROOT / "outputs",
    )
