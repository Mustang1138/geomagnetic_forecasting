"""
Parsing utilities for geomagnetic forecasting project.
CORRECTED VERSION with accurate column positions from OMNI2 format specification.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def parse_omni2_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse a fixed-width OMNI2 data file.

    Column specifications follow the official NASA SPDF OMNI2 format
    description to ensure scientific correctness
    (NASA SPDF, 2024).
    """

    # The selected OMNI2 parameters represent well-established drivers and
    # responses in geomagnetic storm dynamics:
    # - Bz (GSM): primary coupling parameter for solar wind–magnetosphere interaction
    # - Solar wind speed and density: control energy input magnitude
    # - Dst index: standard measure of geomagnetic storm intensity
    # (Abdullah et al., 2022; Zou et al., 2024).

    colspecs = [
        (0, 4),  # Year
        (4, 8),  # DOY
        (8, 11),  # Hour
        (36, 42),  # bt: Magnitude of average field vector |<B>| (word 10, F6.1)
        (78, 84),  # Bz GSM
        (123, 129),  # Proton density
        (129, 135),  # Bulk speed
        (225, 231),  # Dst
    ]

    names = [
        "year",
        "doy",
        "hour",
        "bt",
        "bz_gsm",
        "density",
        "speed",
        "dst",
    ]

    # OMNI2 uses multiple sentinel values across different fixed-width formats
    # (e.g. F6.0, I6), depending on variable and year.
    # Explicit enumeration avoids misclassification of invalid measurements
    # as physical extremes, a known source of ML degradation
    # (NASA SPDF, 2024; Cristoforetti et al., 2022).

    na_values = [
        "9999", " 9999", "999.9", " 999.9", "9999.", " 9999.",
        " 999.9", "999.99", "999999", " 999999", " 99999", "99999",
        "   99", "  9.9", "-999.9", " 999.99", "9999999", "999999.99",
        "99999", " 99999", "9999. ", " 9999. ",  # Added for F6.0/I6 variations
        "999.9 ", " 999.9 ", "9.999", " 9.999"
    ]

    try:
        df = pd.read_fwf(
            filepath,
            colspecs=colspecs,
            names=names,
            na_values=na_values,
            keep_default_na=False,
            header=None,
            infer_nrows=5000,
        )

        for col in names:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["year", "doy", "hour"])

        df["datetime"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" +
            df["doy"].astype(int).astype(str).str.zfill(3) + " " +
            df["hour"].astype(int).astype(str).str.zfill(2) + ":00:00",
            format="%Y-%j %H:%M:%S",
            errors="coerce"
        )

        df = df.dropna(subset=["datetime"])

        out = df[["datetime", "bt", "bz_gsm", "speed", "density", "dst"]].copy()
        out = out.sort_values("datetime").reset_index(drop=True)

        print(
            f"Parsed {
            filepath.name}: {
            len(out)} rows, range {
            out['datetime'].min()} → {
            out['datetime'].max()}")

        return out

    except Exception as err:
        print(f"Failed parsing {filepath.name}: {err}")
        return None


def parse_dscovr_json(data: list) -> Optional[pd.DataFrame]:
    """
    Parse DSCOVR JSON format returned by NOAA SWPC.
    """
    if len(data) < 2:
        return None

    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    df["time_tag"] = pd.to_datetime(df["time_tag"])

    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
