"""Storm-window detail figures: time-series overlay and residual diagnostic."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.figures._common import (
    MODEL_META,
    OUT_DIR,
    STORM_WINDOW,
    load_predictions,
    plot_overlay,
)


def fig_storm_timeseries(preds: dict, output_path: Path) -> None:
    """Storm-window time-series detail (steps 470–580)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_overlay(ax, preds, *STORM_WINDOW, legend=True)
    ax.set_title(
        f"Storm detail — test steps {STORM_WINDOW[0]}–{STORM_WINDOW[1]}  (peak SSI ≈ 0.63)",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Test step", fontsize=9)
    ax.set_ylabel("SSI", fontsize=9)
    ax.set_ylim(-0.02, 0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_storm_residuals(preds: dict, output_path: Path) -> None:
    """Storm-window residual analysis: signed residuals over absolute error."""
    s, e = STORM_WINDOW
    y_obs = preds["rf"][0]

    fig, (ax_signed, ax_abs) = plt.subplots(2, 1, figsize=(11, 6),
                                            gridspec_kw={"hspace": 0.42})

    for label, key, colour in MODEL_META:
        y_pred = preds[key][1]
        offset = len(y_obs) - len(y_pred)
        ls, le = max(0, s - offset), max(0, e - offset)
        ys = y_pred[ls:le]
        x0 = s + max(0, offset - s)
        obs = y_obs[x0:x0 + len(ys)]
        xs = np.arange(x0, x0 + len(ys))
        signed = obs - ys
        ax_signed.plot(xs, signed, label=label, color=colour, linewidth=0.9, alpha=0.9)
        ax_abs.plot(xs, np.abs(signed), label=label, color=colour, linewidth=0.9, alpha=0.9)

    ax_signed.axhline(0, color="#1a1a1a", linewidth=0.8, linestyle="--")
    ax_signed.set_ylim(-0.25, 0.25)
    ax_signed.set_title("Signed residuals (Observed − Predicted)",
                        fontsize=10, fontweight="bold")
    ax_signed.set_ylabel("Residual (SSI)", fontsize=9)
    ax_signed.legend(fontsize=8, ncol=2, loc="upper right")
    ax_signed.tick_params(labelsize=8)

    ax_abs.set_title("Absolute error |Observed − Predicted|",
                     fontsize=10, fontweight="bold")
    ax_abs.set_xlabel("Test step", fontsize=9)
    ax_abs.set_ylabel("|Error| (SSI)", fontsize=9)
    ax_abs.set_ylim(0)
    ax_abs.tick_params(labelsize=8)

    fig.suptitle(f"Residual analysis — storm window (steps {s}–{e})",
                 fontsize=11, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    preds = load_predictions()
    fig_storm_timeseries(preds, OUT_DIR / "fig_storm_timeseries.png")
    fig_storm_residuals(preds, OUT_DIR / "fig_storm_residuals.png")
