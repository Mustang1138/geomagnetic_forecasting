"""
Run all baseline models and generate prediction artefacts.

Consumes frozen preprocessed datasets and produces prediction CSVs
for downstream evaluation.
"""

from src.models.baseline_models import BaselineTrainer
from src.utils import setup_logging

logger = setup_logging()


def main():
    """
    Train baseline regression models (Linear Regression, Random Forest).
    """
    logger.info("Running baseline model training")

    # Non-temporal ML baselines
    trainer = BaselineTrainer()
    trainer.run()

    logger.info("Baseline model training complete")


if __name__ == "__main__":
    main()
