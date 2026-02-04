"""
Unit tests for derived feature construction.

This test module validates the physics-informed feature engineering
functions, ensuring that derived quantities (SSI, severity classes,
auroral latitude) behave as expected and maintain physical plausibility.

The tests use synthetic data to verify mathematical properties and
boundary conditions, following best practices for scientific software
testing (Martin, 2008).

References:
- Burton et al. (1975) - Physical foundations of storm indices
- Gonzalez et al. (1994) - Storm classification systems
- Martin (2008) - Software testing principles
"""

import numpy as np
import pandas as pd

from src.derived_features import (
    compute_storm_severity_index,
    assign_storm_severity_class,
    estimate_auroral_latitude,
)


def _make_fake_df(n: int = 500) -> pd.DataFrame:
    """
    Generate synthetic OMNI-like data for testing derived features.

    This function creates a DataFrame with gradually intensifying storm
    conditions, enabling tests of monotonicity and physical bounds.

    Parameters
    ----------
    n : int, optional
        Number of samples to generate.
        Default is 500.

    Returns
    -------
    pd.DataFrame
        Synthetic DataFrame containing:
        - dst: Linearly decreasing from 0 to -300 nT (quiet → extreme storm)
        - bz_gsm: Linearly decreasing from +5 to -20 nT (northward → southward)
        - speed: Linearly increasing from 300 to 800 km/s (slow → fast)
        - density: Linearly increasing from 1 to 40 particles/cm³ (low → high)

    Notes
    -----
    The linear progressions create ideal test conditions where:
    1. Storm Severity Index should increase monotonically
    2. All normalisation functions receive their full input range
    3. Boundary conditions (min/max values) are tested

    This synthetic dataset is not physically realistic (real storms don't
    evolve linearly), but it provides deterministic test cases for
    validating mathematical properties (Martin, 2008).
    """
    return pd.DataFrame({
        "dst": np.linspace(0, -300, n),  # Quiet → extreme storm
        "bz_gsm": np.linspace(5, -20, n),  # Northward → strongly southward
        "bt": np.linspace(3, 25, n),
        "speed": np.linspace(300, 800, n),  # Slow → fast solar wind
        "density": np.linspace(1, 40, n),  # Low → high density
    })


def test_ssi_bounds():
    """
    Test that Storm Severity Index remains within [0, 1] bounds.

    The SSI is designed as a normalised metric bounded to [0, 1], where:
    - 0.0 represents quiet geomagnetic conditions
    - 1.0 represents extreme storm conditions

    This test verifies that the normalisation and weighting produce
    values strictly within this range, even with extreme input values.

    Assertions
    ----------
    - Minimum SSI >= 0.0 (no negative severity)
    - Maximum SSI <= 1.0 (properly normalised)

    Notes
    -----
    Violating these bounds would indicate an error in normalisation
    logic or weight configuration. The bounds are critical for:
    1. Consistent model training (bounded target space)
    2. Interpretability (percentage-like severity metric)
    3. Downstream categorical binning (severity classes)
    """
    # Generate synthetic data with full parameter ranges
    df = compute_storm_severity_index(_make_fake_df())
    ssi = df["storm_severity_index"]

    # Verify lower bound
    assert ssi.min() >= 0.0, \
        f"SSI minimum ({ssi.min():.4f}) is below 0.0"

    # Verify upper bound
    assert ssi.max() <= 1.0, \
        f"SSI maximum ({ssi.max():.4f}) exceeds 1.0"


def test_ssi_monotonicity():
    """
    Test that SSI increases as storm drivers intensify.

    With synthetic data where all storm drivers (Dst, Bz, speed, density)
    intensify monotonically, the combined SSI should also increase
    monotonically (or remain constant, allowing for numerical precision).

    This validates that:
    1. Normalisation functions behave correctly
    2. Weights are positive (all components contribute positively)
    3. No computational errors introduce non-monotonic artefacts

    Assertions
    ----------
    - SSI differences between consecutive samples are non-negative
      (allowing for small numerical precision errors)

    Notes
    -----
    The tolerance of -1e-6 accounts for floating-point arithmetic precision.
    True violations would be significantly larger than this threshold.

    Monotonicity is expected given:
    - Linear increase in speed and density (Burton et al., 1975)
    - Linear decrease in Dst (more negative = stronger storm)
    - Linear decrease in Bz (more southward = stronger coupling)
    """
    # Generate synthetic data with monotonically intensifying conditions
    df = compute_storm_severity_index(_make_fake_df())
    ssi = df["storm_severity_index"].values

    # Compute differences between consecutive SSI values
    # Should all be non-negative (allowing tiny negative values from rounding)
    diffs = np.diff(ssi)

    # Verify monotonic increase (with numerical tolerance)
    assert np.all(diffs >= -1e-6), \
        f"SSI is not monotonic: found {np.sum(diffs < -1e-6)} decreasing steps"


def test_severity_class_assignment():
    """
    Test that storm severity classes are correctly assigned from SSI.

    This test verifies:
    1. The severity class column is created
    2. No missing values (NaN) exist after binning
    3. All SSI values are successfully mapped to a class

    Assertions
    ----------
    - Column 'storm_severity_class' exists in output DataFrame
    - No NaN values in severity class column

    Notes
    -----
    The severity classes (quiet, minor, moderate, severe, extreme) provide
    a categorical interpretation of the continuous SSI metric, aligned with
    NOAA-style storm scales (Gonzalez et al., 1994).

    Missing values would indicate that some SSI values fall outside the
    expected [0, 1] range or that the binning configuration is incorrect.
    """
    # Generate SSI
    df = compute_storm_severity_index(_make_fake_df())

    # Assign categorical classes
    df = assign_storm_severity_class(df)

    # Verify column exists
    assert "storm_severity_class" in df.columns, \
        "storm_severity_class column not found in DataFrame"

    # Verify no missing values
    n_missing = df["storm_severity_class"].isna().sum()
    assert n_missing == 0, \
        f"Found {n_missing} missing values in storm_severity_class"


def test_auroral_latitude_range():
    """
    Test that estimated auroral latitude remains within physical bounds.

    The auroral oval boundary is estimated to provide a qualitative
    visualisation of storm effects. The estimate should remain within
    physically plausible latitude ranges:
    - Maximum (quiet conditions): ~67° (polar regions)
    - Minimum (extreme storms): ~45° (mid-latitudes)

    Assertions
    ----------
    - Minimum latitude >= 45.0° (extreme storm limit)
    - Maximum latitude <= 67.0° (quiet-time limit)

    Notes
    -----
    These bounds reflect empirical observations of auroral oval expansion
    during geomagnetic storms. During the most intense storms, aurora can
    be visible at mid-latitudes (~45°), whilst quiet conditions confine
    aurora to polar regions (~67°) (Gonzalez et al., 1994).

    Importantly, this estimate is for visualisation only and should NOT
    be interpreted as a precise geophysical prediction. Real auroral
    dynamics depend on many factors beyond the simple SSI metric.
    """
    # Generate SSI
    df = compute_storm_severity_index(_make_fake_df())

    # Estimate auroral latitude
    df = estimate_auroral_latitude(df)

    # Extract latitude estimates
    lat = df["auroral_latitude_deg"]

    # Verify lower bound (extreme storms)
    assert lat.min() >= 45.0, \
        f"Minimum auroral latitude ({lat.min():.2f}°) is below 45.0°"

    # Verify upper bound (quiet conditions)
    assert lat.max() <= 67.0, \
        f"Maximum auroral latitude ({lat.max():.2f}°) exceeds 67.0°"
