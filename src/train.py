"""
Entry point for preprocessing and validation prior to model training.

Separating preprocessing from model training improves modularity,
testability, and reproducibility
(Martin, 2008; Lwakatare et al., 2020).
"""

from src.preprocess import DataPreprocessor
from src.validators import validate_preprocessed_data

# Execute preprocessing pipeline
pre = DataPreprocessor()
summary = pre.run()

# Validate that outputs meet minimum requirements for ML training
validate_preprocessed_data(summary)
