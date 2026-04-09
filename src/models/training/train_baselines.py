"""Entry point for training all non-temporal baseline regression models."""

from src.models.baseline_models import BaselineTrainer
from src.utils import setup_logging

logger = setup_logging()


def main():
    """Train Linear Regression and Random Forest baselines and save prediction artefacts."""
    logger.info("Running baseline model training")
    BaselineTrainer().run()
    logger.info("Baseline model training complete")


if __name__ == "__main__":
    main()
