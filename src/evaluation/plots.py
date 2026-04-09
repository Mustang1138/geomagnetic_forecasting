"""Stateless matplotlib plotting utilities for geomagnetic model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")


def plot_timeseries(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
        n_points: int = 500,
) -> None:
    """Save a time-series chart of predicted versus true SSI.

    Parameters
    ----------
    n_points
        Maximum number of points to plot; limits figure readability at scale.
    """
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


def plot_scatter(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
) -> None:
    """Save a scatter plot of predicted versus true SSI with a perfect-prediction diagonal."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4)

    # Diagonal line represents perfect prediction across the full value range.
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)

    plt.xlabel("True SSI")
    plt.ylabel("Predicted SSI")
    plt.title(f"{model_name}: Predicted vs true SSI")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_ranking(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart ranking all models by RMSE (lower is better)."""
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


def plot_feature_importance(model, feature_names: list, output_path: Path) -> None:
    """Save a horizontal bar chart of feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        # Silently skip models that do not expose feature_importances_.
        return

    importances = model.feature_importances_

    # Guard against a length mismatch when column lists include non-feature columns.
    aligned_names = feature_names[:len(importances)]

    df = pd.DataFrame({
        "feature": aligned_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
) -> None:
    """Save a residual scatter plot with predicted value on the x-axis."""
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.xlabel("Predicted SSI")
    plt.ylabel("Residual (True − Predicted)")
    plt.title(f"{model_name}: Residual plot")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_residual_distribution(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
) -> None:
    """Save a histogram of prediction residuals."""
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual (True − Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{model_name}: Residual distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
