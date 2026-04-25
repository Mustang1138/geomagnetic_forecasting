"""Stateless matplotlib plotting utilities for geomagnetic model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

MODEL_COLOURS = {
    "persistence":        "steelblue",
    "linear_regression":  "mediumseagreen",
    "random_forest":      "darkorange",
    "lstm":               "crimson",
    "gru":                "mediumpurple",
}

MODEL_DISPLAY_NAMES = {
    "persistence":        "Persistence",
    "linear_regression":  "Linear Regression",
    "random_forest":      "Random Forest",
    "lstm":               "LSTM",
    "gru":                "GRU",
}


def plot_timeseries(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
        n_points: int = 500,
        colour: str | None = None,
) -> None:
    """Save a time-series chart of predicted versus true SSI.

    Parameters
    ----------
    n_points
        Maximum number of points to plot; limits figure readability at scale.
    colour
        Line colour for the predicted series. Defaults to the project palette
        entry for model_name, falling back to the matplotlib default.
    """
    colour = colour or MODEL_COLOURS.get(model_name)
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:n_points], label="Observed SSI", color="#1a1a1a",
             linewidth=0.9, linestyle="--")
    plt.plot(y_pred[:n_points], label="Predicted SSI", color=colour,
             linewidth=0.9, alpha=0.9)
    plt.xlabel("Time step")
    plt.ylabel("Storm Severity Index (SSI)")
    plt.title(f"{display}: SSI prediction vs observed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scatter(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
        colour: str | None = None,
) -> None:
    """Save a scatter plot of predicted versus true SSI with a perfect-prediction diagonal."""
    colour = colour or MODEL_COLOURS.get(model_name)
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.3, s=4, color=colour)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.0)

    plt.xlabel("Observed SSI")
    plt.ylabel("Predicted SSI")
    plt.title(f"{display}: Predicted vs observed SSI")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_ranking(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart ranking all models by RMSE (lower is better)."""
    df = metrics_df.sort_values("rmse").copy()
    colours = [MODEL_COLOURS.get(m, "#888888") for m in df["model"]]
    labels  = [MODEL_DISPLAY_NAMES.get(m, m) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, df["rmse"], color=colours)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_ylabel("RMSE (Storm Severity Index)")
    ax.set_xlabel("Model")
    ax.set_title("Model RMSE Comparison — All Five Models (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names: list, output_path: Path) -> None:
    """Save a horizontal bar chart of feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    aligned_names = feature_names[:len(importances)]

    df = pd.DataFrame({
        "feature": aligned_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"],
             color=MODEL_COLOURS.get("random_forest", "darkorange"))
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
        colour: str | None = None,
) -> None:
    """Save a residual scatter plot with predicted value on the x-axis."""
    colour = colour or MODEL_COLOURS.get(model_name)
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.3, s=4, color=colour)
    plt.axhline(0, color="#1a1a1a", linestyle="--", linewidth=1.0)
    plt.xlabel("Predicted SSI")
    plt.ylabel("Residual (Observed − Predicted)")
    plt.title(f"{display}: Residual plot")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ssi_class_distribution(
        train_csv: Path,
        test_csv: Path,
        output_path: Path,
) -> None:
    """Save a grouped bar chart comparing SSI class proportions in training vs test sets.

    Uses the pre-computed ``storm_severity_class`` column written by the
    preprocessing pipeline.  Proportions are shown rather than raw counts so
    that the two splits (which differ substantially in size) are directly
    comparable.
    """
    # Ordered from quietest to most severe so bars read left-to-right.
    class_order = ["quiet", "minor", "moderate", "severe", "extreme"]
    display_labels = ["Quiet\n(< 0.15)", "Minor\n(0.15–0.30)", "Moderate\n(0.30–0.50)",
                      "Severe\n(0.50–0.75)", "Extreme\n(≥ 0.75)"]

    df_train = pd.read_csv(train_csv, usecols=["storm_severity_class"])
    df_test = pd.read_csv(test_csv, usecols=["storm_severity_class"])

    def _proportions(df: pd.DataFrame) -> np.ndarray:
        counts = df["storm_severity_class"].value_counts()
        return np.array([counts.get(cls, 0) / len(df) * 100 for cls in class_order])

    train_pct = _proportions(df_train)
    test_pct = _proportions(df_test)

    x = np.arange(len(class_order))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_train = ax.bar(x - bar_width / 2, train_pct, bar_width,
                        label=f"Training (n={len(df_train):,})", color="steelblue")
    bars_test = ax.bar(x + bar_width / 2, test_pct, bar_width,
                       label=f"Test (n={len(df_test):,})", color="crimson")

    # Annotate each bar with its percentage value.
    for bar in (*bars_train, *bars_test):
        height = bar.get_height()
        if height > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.3,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Proportion of samples (%)")
    ax.set_xlabel("Storm Severity Class (SSI threshold)")
    ax.set_title("SSI Class Distribution: Training Set vs Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_distribution(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_path: Path,
        colour: str | None = None,
) -> None:
    """Save a histogram of prediction residuals."""
    colour = colour or MODEL_COLOURS.get(model_name)
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=50, color=colour, alpha=0.85)
    plt.xlabel("Residual (Observed − Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{display}: Residual distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
