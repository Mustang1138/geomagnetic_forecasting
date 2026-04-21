"""Standalone runner that generates Figure 5.4 for the dissertation.

Figure 5.4: SSI class distribution in the training set versus the test set.

Usage::

    python -m src.evaluation.plot_ssi_distribution

Output: ``outputs/ssi_class_distribution.png``
"""

from pathlib import Path

from src.evaluation.plots import plot_ssi_class_distribution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "ssi_class_distribution.png"


def main() -> None:
    train_csv = PROCESSED_DIR / "train_baseline.csv"
    test_csv = PROCESSED_DIR / "test_baseline.csv"

    for path in (train_csv, test_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python -m src.preprocessing.prepare_data` first."
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_ssi_class_distribution(train_csv, test_csv, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
