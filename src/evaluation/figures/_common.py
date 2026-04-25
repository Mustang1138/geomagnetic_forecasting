"""Shared helpers, paths, and palette constants for the dissertation figures."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
BASELINES_PRED_DIR = OUT_DIR / "baselines" / "predictions"
TEMPORAL_PRED_DIR = OUT_DIR / "temporal" / "predictions"
METRICS_DIR = OUT_DIR / "metrics"

ACCENT = "steelblue"
GREEN = "mediumseagreen"
ORANGE = "darkorange"
RED = "crimson"
PURPLE = "mediumpurple"

MODEL_META = [
    ("Persistence", "pe", ACCENT),
    ("Linear Regression", "lr", GREEN),
    ("Random Forest", "rf", ORANGE),
    ("LSTM", "ls", RED),
    ("GRU", "gr", PURPLE),
]

STORM_WINDOW = (470, 580)
CONTEXT_END = 620


def make_cmap(colour: str) -> LinearSegmentedColormap:
    """Light-to-full gradient colormap matching the project palette."""
    r, g, b = mcolors.to_rgb(colour)
    light = (0.70 + 0.30 * r, 0.70 + 0.30 * g, 0.70 + 0.30 * b)
    return LinearSegmentedColormap.from_list("", [light, colour])


def load_predictions() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{key: (y_true, y_pred)}`` for all five models."""
    rf = pd.read_csv(BASELINES_PRED_DIR / "random_forest_test_predictions.csv")
    lr = pd.read_csv(BASELINES_PRED_DIR / "linear_regression_test_predictions.csv")
    lstm = pd.read_csv(TEMPORAL_PRED_DIR / "lstm_predictions.csv")
    gru = pd.read_csv(TEMPORAL_PRED_DIR / "gru_predictions.csv")

    y_true_full = rf["y_true"].to_numpy()
    y_pred_pe = np.concatenate([[np.nan], y_true_full[:-1]])

    return {
        "pe": (y_true_full[1:], y_pred_pe[1:]),
        "lr": (lr["y_true"].to_numpy(), lr["y_pred"].to_numpy()),
        "rf": (rf["y_true"].to_numpy(), rf["y_pred"].to_numpy()),
        "ls": (lstm["y_true"].to_numpy(), lstm["y_pred"].to_numpy()),
        "gr": (gru["y_true"].to_numpy(), gru["y_pred"].to_numpy()),
    }


def plot_overlay(ax, preds: dict, start: int, end: int, legend: bool = True) -> None:
    """Draw all five model predictions plus the observed series on a single axes."""
    y_obs = preds["rf"][0]
    xs = np.arange(start, min(end, len(y_obs)))
    ax.plot(xs, y_obs[start:end], label="Observed", color="#1a1a1a",
            linewidth=0.9, linestyle="--", zorder=10)
    for label, key, colour in MODEL_META:
        y_pred = preds[key][1]
        offset = len(preds["rf"][0]) - len(y_pred)
        local_start = max(0, start - offset)
        local_end = max(0, end - offset)
        ys = y_pred[local_start:local_end]
        x_start = start + max(0, offset - start)
        ax.plot(np.arange(x_start, x_start + len(ys)), ys,
                label=label, color=colour, linewidth=0.9, alpha=0.85)
    if legend:
        ax.legend(fontsize=8, loc="upper left", ncol=2)
