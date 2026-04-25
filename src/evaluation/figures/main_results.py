"""Headline test-set result figures: combined time-series, scatter, and metric bars."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.figures._common import (
    CONTEXT_END,
    MODEL_META,
    OUT_DIR,
    STORM_WINDOW,
    load_predictions,
    make_cmap,
    plot_overlay,
)
from src.evaluation.plots import MODEL_COLOURS, MODEL_DISPLAY_NAMES


def _grid(nrows: int, ncols: int, figsize: tuple) -> tuple:
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
    return fig, axes


def fig_timeseries_combined(preds: dict, output_path: Path) -> None:
    """Figure 6.1 — full-context and storm-detail time-series for all five models."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.45},
    )

    plot_overlay(ax_top, preds, 0, CONTEXT_END, legend=True)
    ax_top.set_title(f"Full context (test steps 0–{CONTEXT_END})",
                     fontsize=10, fontweight="bold")
    ax_top.set_xlabel("Test step", fontsize=9)
    ax_top.set_ylabel("SSI", fontsize=9)
    ax_top.set_ylim(-0.02, 0.75)
    ax_top.axvspan(*STORM_WINDOW, alpha=0.12, color="gold", label="_nolegend_")

    plot_overlay(ax_bot, preds, *STORM_WINDOW, legend=False)
    ax_bot.set_title(
        f"Storm detail (test steps {STORM_WINDOW[0]}–{STORM_WINDOW[1]}, "
        f"peak SSI ≈ 0.63)",
        fontsize=10, fontweight="bold",
    )
    ax_bot.set_xlabel("Test step", fontsize=9)
    ax_bot.set_ylabel("SSI", fontsize=9)
    ax_bot.set_ylim(-0.02, 0.75)

    fig.suptitle("SSI Time-Series: Observed vs Predicted — All Five Models",
                 fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_scatter_combined(preds: dict, output_path: Path) -> None:
    """Figure 6.2 — predicted vs observed hexbin density, one panel per model."""
    fig, axes = _grid(2, 3, figsize=(15, 7))

    for ax, (label, key, colour) in zip(axes, MODEL_META):
        y_true, y_pred = preds[key]
        hb = ax.hexbin(y_pred, y_true, gridsize=25, cmap=make_cmap(colour),
                       mincnt=1, linewidths=0.2, bins="log")
        ax.plot([0, 0.75], [0, 0.75], "k--", linewidth=1.0)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=8, va="top", fontweight="bold")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted SSI", fontsize=8)
        ax.set_ylabel("Observed SSI", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.01, 0.75)
        ax.set_ylim(-0.01, 0.75)
        fig.colorbar(hb, ax=ax, shrink=0.75, label="log₁₀(Count)", pad=0.02)

    axes[-1].set_visible(False)

    fig.suptitle("Predicted vs Observed SSI (Hexbin Density) — Held-Out Test Set",
                 fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_metrics_rmse_mae(output_path: Path) -> None:
    """Figure 6.3 — grouped bar chart of RMSE and MAE."""
    metrics_df = pd.read_csv(OUT_DIR / "metrics_all_models.csv")
    df = metrics_df.sort_values("rmse")
    colours = [MODEL_COLOURS.get(m, "#888") for m in df["model"]]
    names = [MODEL_DISPLAY_NAMES.get(m, m) for m in df["model"]]
    x, w = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - w / 2, df["rmse"], w, color=colours, label="RMSE")
    bars_m = ax.bar(x + w / 2, df["mae"], w, color=colours, alpha=0.45,
                    hatch="///", label="MAE")
    ax.bar_label(bars_r, fmt="%.4f", padding=2, fontsize=7.5)
    ax.bar_label(bars_m, fmt="%.4f", padding=2, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0)
    ax.set_ylabel("SSI units")
    ax.set_title("RMSE & MAE — All Five Models  (↓ lower is better)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_metrics_r2_ss(output_path: Path) -> None:
    """Figure 6.4 — grouped bar chart of R² and Skill Score with persistence threshold."""
    metrics_df = pd.read_csv(OUT_DIR / "metrics_all_models.csv")
    rmse_pe = metrics_df.loc[metrics_df["model"] == "persistence", "rmse"].values[0]
    metrics_df["skill_score"] = 1 - (metrics_df["rmse"] / rmse_pe) ** 2
    df = metrics_df.sort_values("r2", ascending=False)
    colours = [MODEL_COLOURS.get(m, "#888") for m in df["model"]]
    names = [MODEL_DISPLAY_NAMES.get(m, m) for m in df["model"]]
    x, w = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - w / 2, df["r2"], w, color=colours, label="R²")
    bars_s = ax.bar(x + w / 2, df["skill_score"], w, color=colours, alpha=0.45,
                    hatch="///", label="Skill Score")
    ax.bar_label(bars_r, fmt="%.3f", padding=2, fontsize=7.5)
    ax.bar_label(bars_s, fmt="%.3f", padding=2, fontsize=7.5)
    ax.axhline(0, color="#1a1a1a", linewidth=1.0, linestyle="--",
               label="Persistence threshold (SS = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(-0.12, 1.05)
    ax.set_title("R² & Skill Score — All Five Models  (↑ higher is better)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_comprehensive(preds: dict, output_path: Path) -> None:
    """Single-figure performance summary combining metrics, scatter, and storm detail."""
    metrics_df = pd.read_csv(OUT_DIR / "metrics_all_models.csv")
    rmse_pe = metrics_df.loc[metrics_df["model"] == "persistence", "rmse"].values[0]
    metrics_df["skill_score"] = 1 - (metrics_df["rmse"] / rmse_pe) ** 2

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.32,
                           height_ratios=[1, 1.3])

    ax_rmse_mae = fig.add_subplot(gs[0, 0])
    ax_r2_ss = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])

    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.42)
    ax_ts = fig.add_subplot(gs_right[0])
    ax_res = fig.add_subplot(gs_right[1])

    # --- Top-left: RMSE & MAE ---
    df_rmse = metrics_df.sort_values("rmse")
    colours_rmse = [MODEL_COLOURS.get(m, "#888") for m in df_rmse["model"]]
    names_rmse = [MODEL_DISPLAY_NAMES.get(m, m) for m in df_rmse["model"]]
    x = np.arange(len(names_rmse))
    w = 0.35
    bars_a = ax_rmse_mae.bar(x - w / 2, df_rmse["rmse"], w, color=colours_rmse, label="RMSE")
    bars_b = ax_rmse_mae.bar(x + w / 2, df_rmse["mae"], w, color=colours_rmse,
                             alpha=0.45, hatch="///", label="MAE")
    ax_rmse_mae.bar_label(bars_a, fmt="%.4f", padding=2, fontsize=6)
    ax_rmse_mae.bar_label(bars_b, fmt="%.4f", padding=2, fontsize=6)
    ax_rmse_mae.set_xticks(x)
    ax_rmse_mae.set_xticklabels(names_rmse, rotation=22, ha="right", fontsize=7.5)
    ax_rmse_mae.tick_params(axis="y", labelsize=7.5)
    ax_rmse_mae.set_ylim(bottom=0)
    ax_rmse_mae.set_title("RMSE & MAE  (↓ better)", fontsize=9, fontweight="bold")
    ax_rmse_mae.legend(fontsize=7.5, loc="upper right")

    # --- Top-right: R² & Skill ---
    df_r2 = metrics_df.sort_values("r2", ascending=False)
    colours_r2 = [MODEL_COLOURS.get(m, "#888") for m in df_r2["model"]]
    names_r2 = [MODEL_DISPLAY_NAMES.get(m, m) for m in df_r2["model"]]
    bars_r2 = ax_r2_ss.bar(x - w / 2, df_r2["r2"], w, color=colours_r2, label="R²")
    bars_ss = ax_r2_ss.bar(x + w / 2, df_r2["skill_score"], w, color=colours_r2,
                           alpha=0.45, hatch="///", label="Skill Score")
    ax_r2_ss.bar_label(bars_r2, fmt="%.3f", padding=2, fontsize=6)
    ax_r2_ss.bar_label(bars_ss, fmt="%.3f", padding=2, fontsize=6)
    ax_r2_ss.axhline(0, color="#1a1a1a", linewidth=0.9, linestyle="--",
                     label="Persistence threshold (SS = 0)")
    ax_r2_ss.set_xticks(x)
    ax_r2_ss.set_xticklabels(names_r2, rotation=22, ha="right", fontsize=7.5)
    ax_r2_ss.tick_params(axis="y", labelsize=7.5)
    ax_r2_ss.set_ylim(-0.1, 1.05)
    ax_r2_ss.set_title("R² & Skill Score  (↑ better)", fontsize=9, fontweight="bold")
    ax_r2_ss.legend(fontsize=7.5, loc="lower right")

    # --- Bottom-left: combined scatter ---
    for label, key, colour in MODEL_META:
        y_true, y_pred = preds[key]
        ax_scatter.scatter(y_true, y_pred, alpha=0.18, s=3, color=colour, label=label)
    ax_scatter.plot([0, 0.95], [0, 0.95], "k--", linewidth=0.8)
    ax_scatter.set_xlim(-0.01, 0.95)
    ax_scatter.set_ylim(-0.01, 0.95)
    ax_scatter.set_xlabel("Observed SSI", fontsize=8)
    ax_scatter.set_ylabel("Predicted SSI", fontsize=8)
    ax_scatter.set_title("Predicted vs Observed SSI\n(all models, held-out test set)",
                         fontsize=9, fontweight="bold")
    ax_scatter.legend(fontsize=7.5, markerscale=3, loc="upper left")
    ax_scatter.tick_params(labelsize=7.5)

    # --- Bottom-right top: storm time-series ---
    s, e = STORM_WINDOW
    y_obs = preds["rf"][0]
    ax_ts.plot(np.arange(s, e), y_obs[s:e], label="Observed", color="#1a1a1a",
               linewidth=0.9, linestyle="--", zorder=10)
    for label, key, colour in MODEL_META:
        y_pred = preds[key][1]
        offset = len(y_obs) - len(y_pred)
        ls, le = max(0, s - offset), max(0, e - offset)
        ys = y_pred[ls:le]
        x0 = s + max(0, offset - s)
        ax_ts.plot(np.arange(x0, x0 + len(ys)), ys,
                   label=label, color=colour, linewidth=0.9, alpha=0.9)
    ax_ts.set_title(f"Storm detail — steps {s}–{e}  (peak SSI ≈ 0.63)",
                    fontsize=9, fontweight="bold")
    ax_ts.set_ylabel("SSI", fontsize=8)
    ax_ts.set_ylim(-0.02, 0.75)
    ax_ts.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax_ts.tick_params(labelsize=7.5)

    # --- Bottom-right bottom: residuals ---
    for label, key, colour in MODEL_META:
        y_pred = preds[key][1]
        offset = len(y_obs) - len(y_pred)
        ls, le = max(0, s - offset), max(0, e - offset)
        ys = y_pred[ls:le]
        x0 = s + max(0, offset - s)
        ax_res.plot(np.arange(x0, x0 + len(ys)), y_obs[x0:x0 + len(ys)] - ys,
                    label=label, color=colour, linewidth=0.9, alpha=0.9)
    ax_res.axhline(0, color="#1a1a1a", linewidth=0.8, linestyle="--")
    ax_res.set_title("Residuals (Observed − Predicted)", fontsize=9, fontweight="bold")
    ax_res.set_xlabel("Test step", fontsize=8)
    ax_res.set_ylabel("Residual (SSI)", fontsize=8)
    ax_res.tick_params(labelsize=7.5)

    fig.suptitle("Geomagnetic Storm Severity Prediction — Model Performance Summary",
                 fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    preds = load_predictions()
    fig_timeseries_combined(preds, OUT_DIR / "fig_6_1_timeseries_combined.png")
    fig_scatter_combined(preds, OUT_DIR / "fig_6_2_scatter_combined.png")
    fig_metrics_rmse_mae(OUT_DIR / "fig_metrics_rmse_mae.png")
    fig_metrics_r2_ss(OUT_DIR / "fig_metrics_r2_ss.png")
    fig_comprehensive(preds, OUT_DIR / "fig_comprehensive.png")
