"""Significance and stratified-skill figures: DM heatmap and stratified bars."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.evaluation.figures._common import METRICS_DIR, PLOTS_DIR
from src.evaluation.plots import MODEL_COLOURS, MODEL_DISPLAY_NAMES

MODEL_ORDER_DM = ["persistence", "linear_regression", "random_forest", "lstm", "gru"]

DM_CMAP = LinearSegmentedColormap.from_list(
    "project_p", ["#f1f1f1", "#ffc499", "#ff9a4d", "#e87621", "#aa4b0d"], N=256,
)


def _build_signed_logp_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logp_matrix, winner_matrix)`` indexed by ``MODEL_ORDER_DM``."""
    n = len(MODEL_ORDER_DM)
    logp = np.full((n, n), np.nan)
    winners = np.empty((n, n), dtype=object)
    for _, row in df.iterrows():
        a, b = row["model_a"], row["model_b"]
        if a not in MODEL_ORDER_DM or b not in MODEL_ORDER_DM:
            continue
        i, j = MODEL_ORDER_DM.index(a), MODEL_ORDER_DM.index(b)
        p = max(row["p_value"], 1e-16)
        logp[i, j] = -np.log10(p)
        logp[j, i] = logp[i, j]
        winner = a if row["d_bar"] < 0 else b
        winners[i, j] = winner
        winners[j, i] = winner
    return logp, winners


def _format_display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def fig_dm_heatmap(main_csv: Path, ablation_csv: Path, output_path: Path) -> None:
    """Two-panel pairwise Diebold–Mariano significance matrix (main vs ablation)."""
    df_main = pd.read_csv(main_csv)
    df_ablation = pd.read_csv(ablation_csv)
    logp_main, winners_main = _build_signed_logp_matrix(df_main)
    logp_abl, winners_abl = _build_signed_logp_matrix(df_ablation)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = [_format_display(m) for m in MODEL_ORDER_DM]

    vmax = float(np.nanmax([np.nanmax(logp_main), np.nanmax(logp_abl)]))
    vmin = 0.0

    winner_abbrev = {
        "persistence": "Pers",
        "linear_regression": "LR",
        "random_forest": "RF",
        "lstm": "LSTM",
        "gru": "GRU",
    }
    for ax, logp, winners, title in (
        (axes[0], logp_main, winners_main, "Main (5-feature)"),
        (axes[1], logp_abl, winners_abl, "Ablation (4-feature, Dst withheld)"),
    ):
        mat = np.where(np.isnan(logp), np.nan, logp)
        im = ax.imshow(mat, cmap=DM_CMAP, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=11)
        for i in range(len(MODEL_ORDER_DM)):
            for j in range(len(MODEL_ORDER_DM)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#555555",
                            fontsize=11)
                    continue
                if np.isnan(logp[i, j]):
                    continue
                p_value = 10 ** (-logp[i, j])
                if p_value < 0.01:
                    marker = f"**{winner_abbrev[winners[i, j]]}**"
                elif p_value < 0.05:
                    marker = f"*{winner_abbrev[winners[i, j]]}*"
                else:
                    marker = "ns"
                colour = "white" if logp[i, j] > vmax * 0.55 else "#1a1a1a"
                ax.text(j, i, marker, ha="center", va="center", color=colour,
                        fontsize=10, fontweight="bold")
        ax.set_xlabel("Model B")
        ax.set_ylabel("Model A")

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("−log₁₀(p) (higher = stronger evidence against the null)")
    fig.suptitle(
        "Pairwise Diebold–Mariano two-sided significance "
        "(Harvey-corrected, Newey–West lag 6)",
        fontsize=12, y=1.02,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_stratified_skill(stratified_csv: Path, output_path: Path) -> None:
    """Storm-epoch stratified skill score by SSI class for all models."""
    df = pd.read_csv(stratified_csv)
    df = df[df["experiment"] == "main_5feature"].copy()
    classes = ["Quiet", "Minor", "Moderate", "Severe", "Extreme"]
    models = ["random_forest", "linear_regression", "lstm", "gru"]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(classes))
    width = 0.2

    for idx, model in enumerate(models):
        vals = []
        for cls in classes:
            sub = df[(df["class"] == cls) & (df["model"] == model)]
            vals.append(sub["skill_score"].iloc[0] if not sub.empty else np.nan)
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=_format_display(model),
               color=MODEL_COLOURS[model], edgecolor="#1a1a1a", linewidth=0.4)

    ax.axhline(0, color=MODEL_COLOURS["persistence"], linewidth=1.2,
               linestyle="--", label="Persistence (skill = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("SSI class")
    ax.set_ylabel("Skill score vs class-restricted persistence")
    ax.set_title("Storm-epoch stratified skill (main 5-feature experiment)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    ax.set_ylim(-1.6, 1.05)
    for idx, model in enumerate(models):
        sub = df[(df["class"] == "Extreme") & (df["model"] == model)]
        if not sub.empty:
            val = sub["skill_score"].iloc[0]
            if val < -1.6:
                offset = (idx - (len(models) - 1) / 2) * width
                ax.annotate(
                    f"{val:.2f}",
                    xy=(x[-1] + offset, -1.55),
                    xytext=(x[-1] + offset, -1.45),
                    ha="center", fontsize=7.5, color="#1a1a1a",
                    arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=0.4),
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dm_heatmap(
        main_csv=METRICS_DIR / "dm_test_main.csv",
        ablation_csv=METRICS_DIR / "dm_test_ablation.csv",
        output_path=PLOTS_DIR / "fig_f3_dm_significance_matrix.png",
    )
    fig_stratified_skill(
        stratified_csv=METRICS_DIR / "stratified_combined.csv",
        output_path=PLOTS_DIR / "fig_6_14_stratified_skill.png",
    )
