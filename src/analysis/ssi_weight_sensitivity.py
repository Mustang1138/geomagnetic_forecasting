"""SSI weight sensitivity analysis.

Perturbs each of the five SSI weights (Dst, Bz, Bt, speed, density) by ±0.05
in turn, redistributing the change equally across the other four weights so
the weight tuple still sums to 1.0, then computes the Pearson correlation
between the perturbed SSI series and the canonical formulation on the training
partition. Saves a 10-row CSV summary to outputs/.

Run from project root:

    python -m src.analysis.ssi_weight_sensitivity

Result is referenced from §3.4 of the project report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_WEIGHTS: dict[str, float] = {
    "dst": 0.35,
    "bz": 0.25,
    "bt": 0.20,
    "speed": 0.10,
    "density": 0.10,
}

NORM_COLUMNS: dict[str, str] = {
    "dst": "dst_norm",
    "bz": "bz_norm",
    "bt": "bt_norm",
    "speed": "speed_norm",
    "density": "density_norm",
}

PERTURBATION = 0.05
TRAIN_CSV = Path("data/processed/train_baseline.csv")
OUTPUT_CSV = Path("outputs/ssi_weight_sensitivity.csv")


def perturb_weights(component: str, delta: float) -> dict[str, float]:
    """Return a weight dict with ``component`` shifted by ``delta`` and the
    change redistributed equally across the other four weights so the total
    remains 1.0.
    """
    perturbed = dict(CANONICAL_WEIGHTS)
    perturbed[component] += delta
    redistribution = -delta / (len(CANONICAL_WEIGHTS) - 1)
    for other in perturbed:
        if other != component:
            perturbed[other] += redistribution
    if not np.isclose(sum(perturbed.values()), 1.0):
        raise ValueError(f"Perturbed weights sum to {sum(perturbed.values())}, expected 1.0")
    if any(w < 0 for w in perturbed.values()):
        raise ValueError(f"Perturbation produced negative weight: {perturbed}")
    return perturbed


def compute_ssi(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute SSI as a weighted sum of the pre-normalised component columns."""
    return sum(weights[c] * df[NORM_COLUMNS[c]] for c in CANONICAL_WEIGHTS)


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    canonical_ssi = compute_ssi(df, CANONICAL_WEIGHTS)

    rows: list[dict[str, object]] = []
    for component in CANONICAL_WEIGHTS:
        for delta in (+PERTURBATION, -PERTURBATION):
            perturbed = perturb_weights(component, delta)
            perturbed_ssi = compute_ssi(df, perturbed)
            r = float(canonical_ssi.corr(perturbed_ssi, method="pearson"))
            rows.append(
                {
                    "perturbed_component": component,
                    "delta": delta,
                    "perturbed_weight": round(perturbed[component], 4),
                    "pearson_r_vs_canonical": round(r, 6),
                }
            )

    result = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    min_r = result["pearson_r_vs_canonical"].min()
    max_r = result["pearson_r_vs_canonical"].max()
    print(f"Training partition: n = {len(df)} samples")
    print(f"Perturbations: {len(result)} (5 components × ±{PERTURBATION})")
    print(f"Pearson r range: [{min_r:.6f}, {max_r:.6f}]")
    print(f"Saved: {OUTPUT_CSV}")
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
