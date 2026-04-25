"""Pairwise Diebold-Mariano test with Harvey et al. (1997) small-sample correction.

Loads per-model test-set prediction CSVs, aligns them onto a common time window,
and computes the DM statistic, Harvey-corrected statistic, and two-sided p-value
for every unordered model pair. Runs on both the main 5-feature experiment and
the 4-feature Dst-withheld ablation and writes the results to CSV.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _newey_west_lr_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance of the loss-differential series.

    Uses Bartlett kernel weights w_k = 1 - k / (lag + 1) on autocovariances
    gamma_k for k = 0, ..., lag.
    """
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    n = d.size
    gamma0 = np.dot(d, d) / n
    total = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.dot(d[k:], d[:-k]) / n
        weight = 1.0 - k / (lag + 1.0)
        total += 2.0 * weight * gamma_k
    # Clip to non-negative; HAC estimators can be marginally negative in finite samples.
    return float(max(total, 0.0))


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    lag: int = 6,
    horizon: int = 1,
) -> dict:
    """Compute DM statistic, Harvey-corrected DM*, and two-sided p-value.

    Parameters
    ----------
    y_true : array-like
        Observed target values.
    pred_a, pred_b : array-like
        Competing model predictions. A positive DM statistic means model A has
        higher squared-error loss than model B (i.e. B is the better forecaster).
    lag : int
        Truncation lag for the Newey-West long-run variance estimator.
    horizon : int
        Forecast horizon h. One-step-ahead for this study.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("y_true, pred_a, pred_b must share a common shape")

    e_a = y_true - pred_a
    e_b = y_true - pred_b
    d = e_a ** 2 - e_b ** 2
    n = d.size
    d_bar = float(d.mean())

    lr_var = _newey_west_lr_variance(d, lag=lag)
    if lr_var <= 0.0 or n == 0:
        return {
            "n": n,
            "d_bar": d_bar,
            "lr_variance": lr_var,
            "dm_stat": np.nan,
            "harvey_stat": np.nan,
            "p_value": np.nan,
        }

    dm_stat = d_bar / np.sqrt(lr_var / n)

    # Harvey et al. (1997) correction for finite samples.
    harvey_factor = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    harvey_stat = dm_stat * harvey_factor

    # Two-sided p-value under a Student-t with n-1 degrees of freedom (Harvey).
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(harvey_stat), df=n - 1))

    return {
        "n": n,
        "d_bar": d_bar,
        "lr_variance": lr_var,
        "dm_stat": float(dm_stat),
        "harvey_stat": float(harvey_stat),
        "p_value": float(p_value),
    }


def _load_model_predictions(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Load y_true and y_pred per model from CSV files."""
    out: dict[str, np.ndarray] = {}
    for model, csv_path in paths.items():
        df = pd.read_csv(csv_path)
        out[f"{model}_y_true"] = df["y_true"].to_numpy(dtype=float)
        out[f"{model}_y_pred"] = df["y_pred"].to_numpy(dtype=float)
    return out


def _assemble_aligned_frame(
    baselines_dir: Path,
    temporal_dir: Path,
) -> pd.DataFrame:
    """Return a DataFrame with aligned y_true and per-model predictions.

    Aligns on the common tail window. Baselines cover every test-set row;
    temporal models drop the first `sequence_length` rows. Persistence is
    reconstructed directly as y_true shifted by one step in physical units.
    """
    data = _load_model_predictions(
        {
            "rf": baselines_dir / "random_forest_test_predictions.csv",
            "lr": baselines_dir / "linear_regression_test_predictions.csv",
            "lstm": temporal_dir / "lstm_predictions.csv",
            "gru": temporal_dir / "gru_predictions.csv",
        }
    )

    n_baseline = data["rf_y_true"].size
    n_temporal = data["lstm_y_true"].size
    offset = n_baseline - n_temporal  # sequence_length samples trimmed from baselines

    # Sanity: last rows of baselines and temporal share identical y_true.
    assert np.allclose(
        data["rf_y_true"][offset:], data["lstm_y_true"], atol=1e-10
    ), "Baseline and temporal y_true tails do not align"

    y_true_full = data["rf_y_true"]  # physical-unit observed SSI across full test set
    # Need one step before the first aligned sample to form persistence, so start
    # alignment at index max(offset, 1).
    start = max(offset, 1)
    y_true_aligned = y_true_full[start:]
    persistence_aligned = y_true_full[start - 1 : -1]

    rf_aligned = data["rf_y_pred"][start:]
    lr_aligned = data["lr_y_pred"][start:]

    temporal_start = start - offset  # index into temporal arrays
    lstm_aligned = data["lstm_y_pred"][temporal_start:]
    gru_aligned = data["gru_y_pred"][temporal_start:]

    frame = pd.DataFrame(
        {
            "y_true": y_true_aligned,
            "random_forest": rf_aligned,
            "linear_regression": lr_aligned,
            "lstm": lstm_aligned,
            "gru": gru_aligned,
            "persistence": persistence_aligned,
        }
    )
    return frame


def run_pairwise_dm(
    aligned: pd.DataFrame,
    lag: int = 6,
    horizon: int = 1,
) -> pd.DataFrame:
    """Run pairwise DM test across all ten unordered model pairs."""
    models = [c for c in aligned.columns if c != "y_true"]
    y_true = aligned["y_true"].to_numpy()
    rows = []
    for model_a, model_b in combinations(models, 2):
        result = diebold_mariano(
            y_true,
            aligned[model_a].to_numpy(),
            aligned[model_b].to_numpy(),
            lag=lag,
            horizon=horizon,
        )
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n": result["n"],
                "d_bar": result["d_bar"],
                "dm_stat": result["dm_stat"],
                "harvey_stat": result["harvey_stat"],
                "p_value": result["p_value"],
            }
        )
    return pd.DataFrame(rows)


def run(
    experiment: str,
    baselines_dir: Path,
    temporal_dir: Path,
    output_path: Path,
    lag: int = 6,
) -> pd.DataFrame:
    aligned = _assemble_aligned_frame(baselines_dir, temporal_dir)
    results = run_pairwise_dm(aligned, lag=lag)
    results.insert(0, "experiment", experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"[{experiment}] wrote {len(results)} pairwise DM results to {output_path}")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    outputs = PROJECT_ROOT / "outputs"

    main_results = run(
        experiment="main_5feature",
        baselines_dir=outputs / "baselines" / "predictions",
        temporal_dir=outputs / "temporal" / "predictions",
        output_path=outputs / "metrics" / "dm_test_main.csv",
        lag=6,
    )

    ablation_results = run(
        experiment="ablation_4feature_no_dst",
        baselines_dir=outputs / "ablation" / "baselines" / "predictions",
        temporal_dir=outputs / "ablation" / "temporal" / "predictions",
        output_path=outputs / "metrics" / "dm_test_ablation.csv",
        lag=6,
    )

    combined = pd.concat([main_results, ablation_results], ignore_index=True)
    combined.to_csv(outputs / "metrics" / "dm_test_combined.csv", index=False)
    print(f"\nCombined DM results written to {outputs / 'metrics' / 'dm_test_combined.csv'}")
