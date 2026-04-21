"""Dst-withheld ablation experiment.

Retrains all five models on four input features (Bt, Bz, speed, density)
with Dst excluded, to test whether contemporaneous Dst regression explains
the Random Forest dominance observed in the main experiment.

Outputs: data/processed/ablation/  and  outputs/ablation/
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.evaluate import compute_regression_metrics
from src.models.baseline_models import BaselineTrainer
from src.models.persistence import persistence_forecast
from src.models.training.train_lstm import train_lstm
from src.models.training.train_gru import train_gru
from src.preprocessing.preprocess import DataPreprocessor, TARGET_COL
from src.utils import ensure_dir, setup_logging

logger = setup_logging()

ABLATION_FEATURES: list[str] = ["bt", "bz_gsm", "speed", "density"]
ABLATION_PROCESSED = Path("data/processed/ablation")
ABLATION_OUTPUTS = Path("outputs/ablation")
MAIN_METRICS_CSV = Path("outputs/metrics_all_models.csv")


class AblationPreprocessor(DataPreprocessor):
    """Four-feature preprocessing variant: Dst excluded from model inputs."""
    FEATURE_COLS = ABLATION_FEATURES


class AblationBaselineTrainer(BaselineTrainer):
    """Baseline trainer that reads the 4-feature ablation CSVs."""
    FEATURE_COLS = ABLATION_FEATURES


def run_ablation_preprocessing(input_csv: str = "data/raw/omni2_combined.csv") -> None:
    """Preprocess with 4 features and save to ABLATION_PROCESSED."""
    ensure_dir(str(ABLATION_PROCESSED))
    preprocessor = AblationPreprocessor()
    summary = preprocessor.run(
        input_csv=input_csv,
        output_dir=str(ABLATION_PROCESSED),
    )
    logger.info("Ablation preprocessing complete: %s", summary)


def run_ablation_baselines() -> dict:
    """Train LR and RF on ablation data; save to ABLATION_OUTPUTS/baselines."""
    trainer = AblationBaselineTrainer()
    return trainer.run(
        processed_dir=str(ABLATION_PROCESSED),
        output_dir=str(ABLATION_OUTPUTS / "baselines"),
    )


def run_ablation_temporal() -> None:
    """Train LSTM and GRU on ablation data; n_features auto-detected from array shape."""
    train_lstm(
        data_dir=str(ABLATION_PROCESSED),
        output_dir=str(ABLATION_OUTPUTS / "temporal"),
    )
    train_gru(
        data_dir=str(ABLATION_PROCESSED),
        output_dir=str(ABLATION_OUTPUTS / "temporal"),
    )


def compute_ablation_metrics() -> pd.DataFrame:
    """Collect prediction CSVs, compute metrics + skill scores, save summary CSV."""
    ensure_dir(str(ABLATION_OUTPUTS))

    scaler_y_path = ABLATION_PROCESSED / "scaler_y.pkl"
    with open(scaler_y_path, "rb") as fh:
        scaler_y = pickle.load(fh)

    rows = []

    baseline_pred_dir = ABLATION_OUTPUTS / "baselines" / "predictions"
    if baseline_pred_dir.exists():
        for pred_file in baseline_pred_dir.glob("*_test_predictions.csv"):
            model_name = pred_file.stem.replace("_test_predictions", "")
            df = pd.read_csv(pred_file)
            metrics = compute_regression_metrics(df["y_true"].values, df["y_pred"].values)
            metrics["model"] = model_name
            rows.append(metrics)

    temporal_pred_dir = ABLATION_OUTPUTS / "temporal" / "predictions"
    if temporal_pred_dir.exists():
        for pred_file in temporal_pred_dir.glob("*_predictions.csv"):
            model_name = pred_file.stem.replace("_predictions", "")
            df = pd.read_csv(pred_file)
            metrics = compute_regression_metrics(df["y_true"].values, df["y_pred"].values)
            metrics["model"] = model_name
            rows.append(metrics)

    # Persistence from ablation test set (same y, so same as main experiment)
    test_csv = ABLATION_PROCESSED / "test_baseline.csv"
    df_test = pd.read_csv(test_csv)
    y_scaled = df_test[TARGET_COL].values
    y_true_scaled, y_pred_scaled = persistence_forecast(y_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred_p = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    p_metrics = compute_regression_metrics(y_true, y_pred_p)
    p_metrics["model"] = "persistence"
    rows.append(p_metrics)

    df = pd.DataFrame(rows)
    persistence_rmse = df.loc[df["model"] == "persistence", "rmse"].values[0]
    df["skill_score"] = 1 - (df["rmse"] / persistence_rmse) ** 2
    df = df.sort_values("rmse").reset_index(drop=True)
    df.to_csv(ABLATION_OUTPUTS / "metrics_ablation.csv", index=False)
    logger.info("Ablation metrics:\n%s", df.to_string())
    return df


def plot_skill_comparison(ablation_df: pd.DataFrame) -> None:
    """Grouped bar chart: skill scores for 5-feature vs 4-feature (Dst-withheld)."""
    if not MAIN_METRICS_CSV.exists():
        logger.warning(
            "Main experiment metrics not found at %s — "
            "run `python -m src.evaluation.evaluate` first. "
            "Skipping comparison plot.", MAIN_METRICS_CSV
        )
        return
    main_df = pd.read_csv(MAIN_METRICS_CSV)
    persistence_rmse_main = main_df.loc[main_df["model"] == "persistence", "rmse"].values[0]
    main_df["skill_score"] = 1 - (main_df["rmse"] / persistence_rmse_main) ** 2

    model_order = ["random_forest", "linear_regression", "lstm", "gru", "persistence"]
    labels = {
        "random_forest": "RF",
        "linear_regression": "LR",
        "lstm": "LSTM",
        "gru": "GRU",
        "persistence": "Persistence",
    }

    def _score(df: pd.DataFrame, model: str) -> float:
        row = df.loc[df["model"] == model, "skill_score"]
        if row.empty:
            logger.warning("Model '%s' not found in metrics — using 0.0 for chart.", model)
            return 0.0
        return float(row.values[0])

    main_scores = [_score(main_df, m) for m in model_order]
    ablation_scores = [_score(ablation_df, m) for m in model_order]

    x = np.arange(len(model_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, main_scores, width,
           label="5-feature (Dst included)", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, ablation_scores, width,
           label="4-feature (Dst withheld)", color="#FF9800", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Persistence threshold")
    ax.set_ylabel("Skill Score SS = 1 − (RMSE / RMSE_persistence)²")
    ax.set_title("Figure 6.15: Skill Score Comparison — 5-Feature vs Dst-Withheld Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in model_order])
    ax.legend()
    ax.set_ylim(-0.3, 1.05)
    plt.tight_layout()

    out_path = ABLATION_OUTPUTS / "fig_6_15_skill_comparison_ablation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Skill comparison plot saved → %s", out_path)


def main() -> None:
    """Run the full Dst-withheld ablation pipeline."""
    logger.info("=== Dst-Withheld Ablation Experiment ===")

    if not (ABLATION_PROCESSED / "train_baseline.csv").exists():
        logger.info("Step 1/4: Preprocessing (4 features, Dst withheld)…")
        run_ablation_preprocessing()
    else:
        logger.info("Step 1/4: Ablation preprocessed data already present — skipping.")

    logger.info("Step 2/4: Training baseline models on ablation data…")
    run_ablation_baselines()

    logger.info("Step 3/4: Training temporal models on ablation data…")
    run_ablation_temporal()

    logger.info("Step 4/4: Computing metrics and generating comparison plot…")
    ablation_df = compute_ablation_metrics()
    plot_skill_comparison(ablation_df)

    logger.info("=== Ablation complete. Results → %s ===", ABLATION_OUTPUTS)
    print("\nAblation Metrics (Dst withheld):")
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
