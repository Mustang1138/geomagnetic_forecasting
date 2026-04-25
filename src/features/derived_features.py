"""Derived features for geomagnetic storm forecasting: SSI, storm class, and auroral latitude."""

import numpy as np
import pandas as pd


def _clip_and_normalise(
        x: pd.Series,
        min_val: float,
        max_val: float,
) -> pd.Series:
    """Clip values to physical bounds and apply min-max normalisation.

    Parameters
    ----------
    x : pd.Series
        Input parameter values.
    min_val : float
        Lower physical bound.
    max_val : float
        Upper physical bound.

    Returns
    -------
    pd.Series
        Normalised values in [0, 1].
    """
    clipped_values = x.clip(lower=min_val, upper=max_val)
    return (clipped_values - min_val) / (max_val - min_val)


def normalise_bt(bt: pd.Series) -> pd.Series:
    """Normalise IMF magnitude Bt (nT) to [0, 1]."""
    return _clip_and_normalise(bt, 0.0, 30.0)


def normalise_dst(dst: pd.Series) -> pd.Series:
    """Normalise Dst index to [0, 1], where 1 represents an extreme storm."""
    return _clip_and_normalise(dst.abs(), 0.0, 300.0)


def normalise_bz(bz: pd.Series) -> pd.Series:
    """Normalise southward IMF Bz to [0, 1]; northward values are clipped to zero."""
    southward_bz = bz.clip(upper=0.0).abs()
    return _clip_and_normalise(southward_bz, 0.0, 20.0)


def normalise_speed(speed: pd.Series) -> pd.Series:
    """Normalise solar wind speed (km/s) to [0, 1]."""
    return _clip_and_normalise(speed, 300.0, 800.0)


def normalise_density(density: pd.Series) -> pd.Series:
    """Normalise solar wind proton density (particles/cm³) to [0, 1]."""
    return _clip_and_normalise(density, 1.0, 50.0)


def compute_storm_severity_index(
        df: pd.DataFrame,
        weights: tuple[float, float, float, float, float] = (0.35, 0.25, 0.20, 0.10, 0.10),
) -> pd.DataFrame:
    """Compute a continuous Storm Severity Index (SSI) in [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns: ``dst``, ``bz_gsm``, ``bt``,
        ``speed``, ``density``.
    weights : tuple of float, optional
        Weights for (Dst, Bz, Bt, speed, density). Must sum to 1.0.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with normalised component columns and
        a ``storm_severity_index`` column appended.

    Raises
    ------
    ValueError
        If the provided weights do not sum to 1.0 (within numerical precision).
    """
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("SSI weights must sum to 1.0")

    dst_weight, bz_weight, bt_weight, speed_weight, density_weight = weights

    df = df.copy()

    df["dst_norm"] = normalise_dst(df["dst"])
    df["bz_norm"] = normalise_bz(df["bz_gsm"])
    df["bt_norm"] = normalise_bt(df["bt"])
    df["speed_norm"] = normalise_speed(df["speed"])
    df["density_norm"] = normalise_density(df["density"])

    df["storm_severity_index"] = (
        dst_weight * df["dst_norm"]
        + bz_weight * df["bz_norm"]
        + bt_weight * df["bt_norm"]
        + speed_weight * df["speed_norm"]
        + density_weight * df["density_norm"]
    )

    return df


def assign_storm_severity_class(df: pd.DataFrame) -> pd.DataFrame:
    """Assign categorical storm severity labels based on SSI.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``storm_severity_index`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with a ``storm_severity_class`` column
        appended. Classes: quiet (0.00–0.15), minor (0.15–0.30),
        moderate (0.30–0.50), severe (0.50–0.75), extreme (0.75–1.00).
    """
    df = df.copy()

    bins = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]
    labels = ["quiet", "minor", "moderate", "severe", "extreme"]

    # include_lowest=True ensures SSI=0.0 falls into the "quiet" category.
    df["storm_severity_class"] = pd.cut(
        df["storm_severity_index"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return df


def estimate_auroral_latitude(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the equatorward auroral oval boundary latitude from SSI.

    For visualisation only, not precise geophysical prediction.
    """
    df = df.copy()

    raw_auroral_latitude = 67.0 - 22.0 * df["storm_severity_index"]
    df["auroral_latitude_deg"] = np.clip(raw_auroral_latitude, 45.0, 67.0)

    return df


def add_all_derived_features(
        df: pd.DataFrame,
        ssi_weights: tuple[float, float, float, float, float] = (0.35, 0.25, 0.20, 0.10, 0.10),
) -> pd.DataFrame:
    """Compute and append SSI, storm severity class, and auroral latitude."""
    df = compute_storm_severity_index(df, weights=ssi_weights)
    df = assign_storm_severity_class(df)
    df = estimate_auroral_latitude(df)
    return df
