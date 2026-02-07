"""
Run all baseline models and generate prediction artefacts.

Consumes frozen preprocessed datasets and produces prediction CSVs
for downstream evaluation.
"""

from pathlib import Path

from src.models.baseline_models import BaselineTrainer
from src.models.persistence import run_persistence_baseline


def main():
    project_root = Path(__file__).resolve().parents[1]

    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"

    # Non-temporal ML baselines
    trainer = BaselineTrainer()
    trainer.run(
        processed_dir=processed_dir,
        output_dir=results_dir / "baselines",
    )

    # Persistence baseline
    run_persistence_baseline(
        processed_dir=processed_dir,
        results_dir=results_dir,
        target_col="storm_severity_index",
    )


if __name__ == "__main__":
    main()
