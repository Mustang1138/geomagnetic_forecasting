"""Storm-epoch stratified metrics.

Bins the aligned test-set predictions by observed SSI class using the §3.4
thresholds (Quiet, Minor, Moderate, Severe, Extreme), then computes RMSE,
MAE, R^2, and a class-restricted skill score against the persistence baseline
within each bin. Writes the results to outputs/metrics/stratified_metrics.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.dm_test import _assemble_aligned_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SSI class thresholds (§3.4).
CLASS_BINS = [-np.inf, 0.15, 0.30, 0.50, 0.75, np.inf]
CLASS_LABELS = ["Quiet", "Minor", "Moderate", "Severe", "Extreme"]

MODEL_ORDER = ["random_forest", "linear_regression", "lstm", "gru", "persistence"]


def _classify(y_true: np.ndarray) -> pd.Series:
    return pd.Series(
        pd.cut(y_true, bins=CLASS_BINS, labels=CLASS_LABELS, right=False)
    )


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.var(y_true) == 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def compute_stratified(aligned: pd.DataFrame) -> pd.DataFrame:
    """Compute per-class metrics for every model plus overall rows."""
    classes = _classify(aligned["y_true"].to_numpy())
    rows: list[dict] = []

    for class_label in CLASS_LABELS + ["Overall"]:
        if class_label == "Overall":
            mask = np.ones(len(aligned), dtype=bool)
        else:
            mask = (classes == class_label).to_numpy()
        n = int(mask.sum())
        if n == 0:
            for model in MODEL_ORDER:
                rows.append(
                    {
                        "class": class_label,
                        "model": model,
                        "n": 0,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "r2": np.nan,
                        "skill_score": np.nan,
                    }
                )
            continue

        y_true = aligned.loc[mask, "y_true"].to_numpy()
        persistence_pred = aligned.loc[mask, "persistence"].to_numpy()
        persistence_mse = float(mean_squared_error(y_true, persistence_pred))

        for model in MODEL_ORDER:
            y_pred = aligned.loc[mask, model].to_numpy()
            mse = float(mean_squared_error(y_true, y_pred))
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = _safe_r2(y_true, y_pred)
            if model == "persistence":
                skill = 0.0
            elif persistence_mse <= 0:
                skill = np.nan
            else:
                skill = 1.0 - mse / persistence_mse
            rows.append(
                {
                    "class": class_label,
                    "model": model,
                    "n": n,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "skill_score": skill,
                }
            )

    return pd.DataFrame(rows)


def run(
    experiment: str,
    baselines_dir: Path,
    temporal_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    aligned = _assemble_aligned_frame(baselines_dir, temporal_dir)
    results = compute_stratified(aligned)
    results.insert(0, "experiment", experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"[{experiment}] wrote stratified metrics to {output_path}")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    outputs = PROJECT_ROOT / "outputs"
    main_results = run(
        experiment="main_5feature",
        baselines_dir=outputs / "baselines" / "predictions",
        temporal_dir=outputs / "temporal" / "predictions",
        output_path=outputs / "metrics" / "stratified_main.csv",
    )
    ablation_results = run(
        experiment="ablation_4feature_no_dst",
        baselines_dir=outputs / "ablation" / "baselines" / "predictions",
        temporal_dir=outputs / "ablation" / "temporal" / "predictions",
        output_path=outputs / "metrics" / "stratified_ablation.csv",
    )
    combined = pd.concat([main_results, ablation_results], ignore_index=True)
    combined.to_csv(outputs / "metrics" / "stratified_combined.csv", index=False)
    print(f"\nCombined stratified metrics written to "
          f"{outputs / 'metrics' / 'stratified_combined.csv'}")
