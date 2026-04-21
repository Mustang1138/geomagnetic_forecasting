"""Verify the ablation preprocessor produces 4-feature arrays."""
from src.evaluation.run_ablation import AblationPreprocessor, ABLATION_FEATURES


def test_ablation_preprocessor_feature_cols():
    """AblationPreprocessor must declare exactly 4 features, excluding dst."""
    assert AblationPreprocessor.FEATURE_COLS == ["bt", "bz_gsm", "speed", "density"]
    assert "dst" not in AblationPreprocessor.FEATURE_COLS
    assert len(AblationPreprocessor.FEATURE_COLS) == 4


def test_ablation_baseline_trainer_feature_cols():
    """AblationBaselineTrainer must use the same 4-feature list."""
    from src.evaluation.run_ablation import AblationBaselineTrainer
    assert AblationBaselineTrainer.FEATURE_COLS == ABLATION_FEATURES
