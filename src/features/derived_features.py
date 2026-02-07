"""
Derived feature construction for geomagnetic storm forecasting.

This module defines physically motivated derived quantities used
as machine learning targets and analysis variables, including:

- Continuous Storm Severity Index (SSI)
- Categorical storm severity classes
- Estimated equatorward auroral oval boundary latitude

All derivations are intentionally simple, interpretable, and grounded
in established space weather literature.

Design principles:
- No data leakage
- Physically meaningful transformations
- Bounded, ML-friendly targets
- Fully reproducible and tunable

References:
- Burton et al. (1975) - Dst index and ring current dynamics
- Gonzalez et al. (1994) - Geomagnetic storm classification
- Liemohn et al. (2021) - Data-model comparison methodologies
"""

from typing import Tuple

import numpy as np
import pandas as pd


# Normalisation utilities (physics-aware)

def _clip_and_normalise(
        x: pd.Series,
        min_val: float,
        max_val: float,
) -> pd.Series:
    """
    Clip values to physical bounds and apply min-max normalisation.

    This function enforces physically meaningful ranges before normalisation,
    preventing unrealistic extreme values from dominating the model training.
    Min-max scaling maps values to [0, 1], making them directly comparable
    and suitable for weighted aggregation.

    Parameters
    ----------
    x : pd.Series
        Input parameter values.
    min_val : float
        Lower physical bound (e.g., minimum observed Dst during quiet conditions).
    max_val : float
        Upper physical bound (e.g., maximum Dst during extreme storms).

    Returns
    -------
    pd.Series
        Normalised values in [0, 1].

    Notes
    -----
    Clipping is applied before normalisation to handle outliers gracefully.
    This approach is preferable to removing outliers, as it preserves sample
    count whilst preventing distortion of the normalised distribution.
    """
    # Clip to physically meaningful range
    x_clipped = x.clip(lower=min_val, upper=max_val)

    # Apply min-max normalisation to map to [0, 1]
    return (x_clipped - min_val) / (max_val - min_val)


def normalise_bt(bt: pd.Series) -> pd.Series:
    """
    Normalise IMF magnitude Bt (nT).

    Bt represents the total available magnetic energy in the solar wind.
    """
    return _clip_and_normalise(bt, 0.0, 30.0)


def normalise_dst(dst: pd.Series) -> pd.Series:
    """
    Normalise Dst index for storm severity.

    The Disturbance Storm Time (Dst) index quantifies the intensity of the
    magnetospheric ring current. Dst is negative during geomagnetic storms;
    larger negative values indicate more intense storms (Burton et al., 1975).

    Parameters
    ----------
    dst : pd.Series
        Dst index values in nanoTesla (nT).

    Returns
    -------
    pd.Series
        Normalised Dst in [0, 1], where 0 = quiet conditions, 1 = extreme storm.

    Notes
    -----
    Physical bounds:
    - Minimum (quiet): 0 nT (no disturbance)
    - Maximum (extreme storm): -300 nT (severe ring current enhancement)

    These bounds are based on NOAA's geomagnetic storm scale and historical
    storm statistics (Gonzalez et al., 1994).
    """
    # Storm-relevant physical bounds (nT)
    # Dst is negative during storms; we use absolute value for normalisation
    return _clip_and_normalise(dst.abs(), 0.0, 300.0)


def normalise_bz(bz: pd.Series) -> pd.Series:
    """
    Normalise southward IMF Bz (GSM coordinates).

    The Interplanetary Magnetic Field (IMF) Bz component in Geocentric Solar
    Magnetospheric (GSM) coordinates is a primary driver of geomagnetic activity.
    Only negative (southward) Bz contributes to geomagnetic disturbances through
    magnetic reconnection at the magnetopause (Burton et al., 1975).

    Parameters
    ----------
    bz : pd.Series
        IMF Bz in GSM coordinates (nT).

    Returns
    -------
    pd.Series
        Normalised southward Bz in [0, 1], where 0 = northward/neutral,
        1 = strongly southward.

    Notes
    -----
    Physical bounds:
    - Minimum (no reconnection): 0 nT (northward or zero field)
    - Maximum (strong reconnection): -20 nT (intense southward field)

    Northward Bz (positive values) is clipped to 0, as it does not contribute
    to geomagnetic activity in the same manner as southward Bz.
    """
    # Only negative (southward) Bz contributes to geomagnetic activity
    # Clip positive values to zero and take absolute value
    bz_southward = bz.clip(upper=0.0).abs()

    # Normalise to [0, 1] using typical storm-time range
    return _clip_and_normalise(bz_southward, 0.0, 20.0)


def normalise_speed(speed: pd.Series) -> pd.Series:
    """
    Normalise solar wind speed.

    Solar wind speed influences the rate of energy transfer from the solar
    wind to the magnetosphere. Higher speeds generally correlate with more
    intense geomagnetic activity, particularly when combined with southward
    IMF (Burton et al., 1975).

    Parameters
    ----------
    speed : pd.Series
        Solar wind speed in km/s.

    Returns
    -------
    pd.Series
        Normalised speed in [0, 1], where 0 = slow wind, 1 = fast wind.

    Notes
    -----
    Physical bounds:
    - Minimum (slow solar wind): 300 km/s (typical quiet-time speed)
    - Maximum (high-speed stream/CME): 800 km/s (storm-associated speeds)

    These bounds encompass the typical range observed in OMNI data
    (Papitashvili and King, 2005, 2020).
    """
    # Typical solar wind speed range during storms
    return _clip_and_normalise(speed, 300.0, 800.0)


def normalise_density(density: pd.Series) -> pd.Series:
    """
    Normalise solar wind proton density.

    Solar wind density affects the dynamic pressure on the magnetosphere.
    Enhanced density, particularly during coronal mass ejections (CMEs),
    can intensify geomagnetic storms (Burton et al., 1975; Gonzalez et al., 1994).

    Parameters
    ----------
    density : pd.Series
        Solar wind proton density in particles/cm³.

    Returns
    -------
    pd.Series
        Normalised density in [0, 1], where 0 = low density, 1 = high density.

    Notes
    -----
    Physical bounds:
    - Minimum (quiet solar wind): 1 particles/cm³
    - Maximum (dense CME sheath): 50 particles/cm³

    These bounds are based on typical OMNI observations and storm studies
    (Papitashvili and King, 2005, 2020).
    """
    # Typical solar wind density range
    return _clip_and_normalise(density, 1.0, 50.0)


# Storm Severity Index (continuous regression target)

def compute_storm_severity_index(
        df: pd.DataFrame,
        weights: Tuple[float, float, float, float, float] = (0.35, 0.25, 0.20, 0.10, 0.10)
) -> pd.DataFrame:
    """
        Compute a continuous Storm Severity Index (SSI) in [0, 1].

        The Storm Severity Index (SSI) aggregates key solar wind and geomagnetic
        drivers into a single continuous metric suitable for regression modelling.
        This provides a more nuanced target than binary storm/no-storm classification
        and allows models to learn the continuous nature of geomagnetic storm intensity.

        The SSI combines five physically motivated parameters:
            1. Dst (ring current strength) — primary storm intensity indicator
            2. IMF Bz (GSM) — controls magnetic reconnection efficiency
            3. IMF Bt (total magnetic field magnitude) — available magnetic energy
            4. Solar wind speed — governs energy transfer rate
            5. Solar wind proton density — contributes to dynamic pressure

        These parameters are grounded in established solar wind–magnetosphere
        coupling theory and empirical storm models
        (Burton et al., 1975; Gonzalez et al., 1994; Papitashvili & King, 2020).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing:
                - 'dst' : Disturbance Storm Time index (nT)
                - 'bz_gsm' : IMF Bz in GSM coordinates (nT)
                - 'bt' : IMF total magnetic field magnitude (nT)
                - 'speed' : Solar wind bulk speed (km/s)
                - 'density' : Solar wind proton density (particles/cm³)

        weights : tuple of float, optional
            Weights for (Dst, Bz, Bt, speed, density).
            Must sum to 1.0.
            Default values reflect the relative physical importance of each
            parameter, with Dst and Bz receiving the highest weights.

        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with added columns:
                - dst_norm : Normalised Dst contribution [0, 1]
                - bz_norm : Normalised southward Bz contribution [0, 1]
                - bt_norm : Normalised Bt contribution [0, 1]
                - speed_norm : Normalised solar wind speed [0, 1]
                - density_norm : Normalised solar wind density [0, 1]
                - storm_severity_index : Weighted SSI in [0, 1]

        Raises
        ------
        ValueError
            If the provided weights do not sum to 1.0 (within numerical precision).

        Notes
        -----
        The SSI is computed entirely in physical space (prior to any statistical
        standardisation) to preserve interpretability. Each component is
        independently normalised before weighted aggregation, preventing any
        single parameter from dominating due to scale differences.

        This design follows best practices for physics-informed feature engineering
        in space weather forecasting (Liemohn et al., 2021).
        """

    # Validate weights sum to 1.0
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("SSI weights must sum to 1.0")

    # Unpack weights for clarity
    w_dst, w_bz, w_bt, w_speed, w_density = weights

    # Create copy to avoid modifying original DataFrame
    df = df.copy()

    # Normalise each physical component to [0, 1]
    # This ensures all parameters contribute on the same scale
    df["dst_norm"] = normalise_dst(df["dst"])
    df["bz_norm"] = normalise_bz(df["bz_gsm"])
    df["bt_norm"] = normalise_bt(df["bt"])
    df["speed_norm"] = normalise_speed(df["speed"])
    df["density_norm"] = normalise_density(df["density"])

    # Compute weighted aggregation
    # The weighting scheme reflects the relative importance of each parameter:
    # - Dst (0.4): Primary indicator of storm intensity
    # - Bz (0.3): Key driver of magnetospheric coupling
    # - Bt
    # - Speed (0.2): Influences energy transfer rate
    # - Density (0.1): Secondary contribution to dynamic pressure
    df["storm_severity_index"] = (
            w_dst * df["dst_norm"]
            + w_bz * df["bz_norm"]
            + w_bt * df["bt_norm"]
            + w_speed * df["speed_norm"]
            + w_density * df["density_norm"]
    )

    return df


# Severity class labels (categorical interpretation)

def assign_storm_severity_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign categorical storm severity labels based on SSI.

    This function maps the continuous SSI to discrete severity classes,
    providing an interpretable categorical view of storm intensity.
    The classes are aligned with NOAA-style geomagnetic storm interpretation
    whilst remaining ML-agnostic (Gonzalez et al., 1994).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'storm_severity_index' column.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added 'storm_severity_class' column.

    Notes
    -----
    Classes:
        - quiet (SSI: 0.00–0.15): No significant geomagnetic activity
        - minor (SSI: 0.15–0.30): Weak storms, minimal impact
        - moderate (SSI: 0.30–0.50): Noticeable auroral activity
        - severe (SSI: 0.50–0.75): Major storm, possible infrastructure effects
        - extreme (SSI: 0.75–1.00): Intense storm, significant impacts

    These thresholds are loosely based on NOAA's G-scale for geomagnetic
    storms, adapted for the continuous SSI metric. The categorisation
    enables both regression (predicting continuous SSI) and classification
    (predicting discrete classes) approaches.
    """
    # Create copy to avoid modifying original DataFrame
    df = df.copy()

    # Define bin edges for SSI thresholds
    # These are informed by NOAA storm scales but adapted to SSI range
    bins = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]

    # Define corresponding severity labels
    labels = ["quiet", "minor", "moderate", "severe", "extreme"]

    # Assign categories using pandas cut function
    # include_lowest=True ensures SSI=0.0 falls into "quiet" category
    df["storm_severity_class"] = pd.cut(
        df["storm_severity_index"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return df


# Auroral latitude proxy (location estimation)

def estimate_auroral_latitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the equatorward auroral oval boundary latitude (degrees).

    During geomagnetic storms, the auroral oval expands equatorward,
    enabling aurora visibility at lower latitudes. This function provides
    a simple empirical estimate of the equatorward boundary based on SSI.

    Stronger geomagnetic storms drive auroral activity equatorward, with
    extreme storms potentially bringing aurora to mid-latitudes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'storm_severity_index' column.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added 'auroral_latitude_deg' column.

    Notes
    -----
    Approximate mapping:
        SSI ~ 0.0  → ~67° (quiet conditions, high-latitude aurora)
        SSI ~ 1.0  → ~45° (extreme storm, mid-latitude aurora)

    This linear relationship is a simplified approximation. In reality,
    auroral dynamics are complex and depend on multiple factors including
    solar wind conditions, magnetospheric configuration, and local time.

    **Important**: This value is intended for visualisation and qualitative
    analysis, NOT precise geophysical prediction. For accurate auroral
    forecasting, dedicated models incorporating additional physics are required.

    The coefficients (67° baseline, 22° range) are based on empirical
    observations of auroral oval expansion during storms but should be
    considered rough estimates (Gonzalez et al., 1994).
    """
    # Create copy to avoid modifying original DataFrame
    df = df.copy()

    # Linear mapping from SSI to latitude
    # Baseline: 67° (typical quiet-time auroral oval position)
    # Range: 22° (maximum equatorward expansion during extreme storms)
    auroral_lat = 67.0 - 22.0 * df["storm_severity_index"]

    # Ensure values remain within physically reasonable bounds
    # Lower bound: 45° (extreme storms rarely push aurora beyond this)
    # Upper bound: 67° (quiet-time polar oval position)
    df["auroral_latitude_deg"] = np.clip(auroral_lat, 45.0, 67.0)

    return df


# Convenience wrapper

def add_all_derived_features(
        df: pd.DataFrame,
        ssi_weights: Tuple[float, float, float, float, float] = (0.30, 0.25, 0.20, 0.15, 0.10),
) -> pd.DataFrame:
    """
    Compute and append all physics-informed derived features.

    This convenience wrapper applies the complete derived-feature pipeline:
        1. Storm Severity Index (SSI) — continuous regression target
        2. Storm severity class — categorical interpretation of SSI
        3. Auroral latitude estimate — qualitative geophysical proxy

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw OMNI parameters:
            - dst, bz_gsm, bt, speed, density

    ssi_weights : tuple of float, optional
        Weights for SSI computation in the order:
        (Dst, Bz, Bt, speed, density).

    Returns
    -------
    pd.DataFrame
        DataFrame with all derived features appended.

    Notes
    -----
    This function must be applied **before feature standardisation**
    in the preprocessing pipeline. All derived quantities are computed
    in physical units to preserve interpretability and scientific meaning.
    """
    # Compute continuous Storm Severity Index and normalised components
    df = compute_storm_severity_index(df, weights=ssi_weights)

    # Assign categorical severity classes based on SSI
    df = assign_storm_severity_class(df)

    # Estimate auroral oval boundary latitude
    df = estimate_auroral_latitude(df)

    return df
