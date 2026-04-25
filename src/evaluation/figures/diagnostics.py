"""Supporting diagnostic figures: solar cycle, SSI ACF, Dst-vs-SSI, class distribution."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.figures._common import (
    ACCENT,
    DATA_DIR,
    GREEN,
    ORANGE,
    OUT_DIR,
    RED,
    make_cmap,
)
from src.evaluation.plots import plot_ssi_class_distribution


def fig_solar_cycle_timeline(output_path: Path) -> None:
    """Schematic solar-cycle activity with chronological train/val/test split shading."""
    years = np.arange(1996, 2027.1, 0.1)

    def _gaussian_cycle(peak_year: float, amplitude: float, width: float = 3.0) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((years - peak_year) / width) ** 2)

    ssn = (
        _gaussian_cycle(2000.5, 170, 2.8)
        + _gaussian_cycle(2003.5, 80, 2.0)
        + _gaussian_cycle(2014.0, 115, 2.5)
        + _gaussian_cycle(2024.5, 200, 2.5)
        + _gaussian_cycle(2019.5, 5, 1.5)
    )
    ssn = np.clip(ssn, 0, None)

    train_start = 2000.0
    train_end = 2018.90
    val_end = 2022.95
    test_end = 2026.99

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(years, ssn, alpha=0.18, color=ACCENT)
    ax.plot(years, ssn, color=ACCENT, linewidth=1.4, label="Approx. solar activity")

    shade_alpha = 0.13
    ax.axvspan(train_start, train_end, alpha=shade_alpha, color=GREEN,
               label="Training (2000–2018)")
    ax.axvspan(train_end, val_end, alpha=shade_alpha, color=ORANGE,
               label="Validation (2018–2022)")
    ax.axvspan(val_end, test_end, alpha=shade_alpha, color=RED,
               label="Test (2022–2026)")

    for x, col in [(train_end, GREEN), (val_end, ORANGE), (test_end, RED)]:
        ax.axvline(x, color=col, linewidth=1.2, linestyle="--", alpha=0.9)

    for cy, yr, ssn_pk in [("SC 23", 2000.5, 178), ("SC 24", 2014.0, 123),
                           ("SC 25", 2024.5, 208)]:
        ax.text(yr, ssn_pk + 12, cy, ha="center", va="bottom",
                fontsize=8.5, color="#333333", fontfamily="monospace")

    ax.set_xlim(1999.5, 2027.0)
    ax.set_ylim(0, 260)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Approx. solar activity (arb.)", fontsize=10)
    ax.set_title(
        "Solar cycle activity and chronological train / validation / test split boundaries",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_ssi_acf(output_path: Path) -> None:
    """SSI autocorrelation function over the training set, lags 0–96 (≈ 24 days)."""
    train_df = pd.read_csv(DATA_DIR / "train_baseline.csv", parse_dates=["datetime"])
    ssi = train_df["storm_severity_index"].dropna().values

    max_lag = 96
    acf = np.array([
        np.corrcoef(ssi[:-lag], ssi[lag:])[0, 1] if lag > 0 else 1.0
        for lag in range(max_lag + 1)
    ])
    lags = np.arange(max_lag + 1)

    n = len(ssi)
    ci = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(lags, -ci, ci, alpha=0.15, color=ACCENT, label="95% CI (white noise)")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    _, stemlines, _ = ax.stem(lags, acf, linefmt="-", markerfmt=" ", basefmt=" ")
    plt.setp(stemlines, color=ACCENT, linewidth=0.9, alpha=0.7)

    ax.plot(1, acf[1], "o", color=ORANGE, zorder=5,
            label=f"Lag 1 (6 h) r = {acf[1]:.3f}")

    ax.set_xlabel("Lag (6-hour steps)", fontsize=10)
    ax.set_ylabel("Autocorrelation", fontsize=10)
    ax.set_title(
        "SSI autocorrelation function — training set (lags 0–96, up to 24 days)",
        fontsize=10, pad=8,
    )
    ax.set_xlim(-1, max_lag + 1)
    ax.set_ylim(-0.25, 1.05)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_dst_vs_ssi(output_path: Path) -> None:
    """Hexbin scatter of normalised Dst against SSI with OLS line and Pearson r."""
    train_df = pd.read_csv(DATA_DIR / "train_baseline.csv", parse_dates=["datetime"])
    test_df = pd.read_csv(DATA_DIR / "test_baseline.csv", parse_dates=["datetime"])
    df = pd.concat([train_df, test_df], ignore_index=True).dropna(
        subset=["dst_norm", "storm_severity_index"]
    )

    dst = df["dst_norm"].values
    ssi = df["storm_severity_index"].values
    r = np.corrcoef(dst, ssi)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(dst, ssi, gridsize=50, cmap=make_cmap(ACCENT),
                   mincnt=1, linewidths=0.2, bins="log")
    fig.colorbar(hb, ax=ax, shrink=0.85, label="log₁₀(Count)", pad=0.02)

    m, b = np.polyfit(dst, ssi, 1)
    x_range = np.linspace(dst.min(), dst.max(), 200)
    ax.plot(x_range, m * x_range + b, color=ORANGE, linewidth=1.6,
            label=f"OLS fit  (r = {r:.3f})")

    ax.set_xlabel("Normalised Dst (input feature)", fontsize=10)
    ax.set_ylabel("Storm Severity Index (target)", fontsize=10)
    ax.set_title("Normalised Dst vs SSI — training and test sets", fontsize=10, pad=8)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_ssi_class_distribution(output_path: Path) -> None:
    """SSI severity-class distribution across train and test partitions."""
    train_csv = DATA_DIR / "train_baseline.csv"
    test_csv = DATA_DIR / "test_baseline.csv"
    for path in (train_csv, test_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python -m src.preprocessing.prepare_data` first."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_ssi_class_distribution(train_csv, test_csv, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    fig_solar_cycle_timeline(OUT_DIR / "fig_solar_cycle_timeline.png")
    fig_ssi_acf(OUT_DIR / "fig_ssi_acf.png")
    fig_dst_vs_ssi(OUT_DIR / "fig_dst_vs_ssi.png")
    fig_ssi_class_distribution(OUT_DIR / "ssi_class_distribution.png")
