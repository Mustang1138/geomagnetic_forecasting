"""Parsing utilities for OMNI2 and DSCOVR data files."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def parse_omni2_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Parse a fixed-width OMNI2 annual data file downloaded from NASA SPDF.

    Parameters
    ----------
    filepath : Path
        Path to the ``.dat`` file.

    Returns
    -------
    pd.DataFrame or None
        Parsed DataFrame sorted by datetime, or ``None`` if parsing fails.
    """
    colspecs = [
        (0, 4),    # Year
        (4, 8),    # DOY
        (8, 11),   # Hour
        (36, 42),  # bt: Magnitude of average field vector |<B>| (F6.1)
        (78, 84),  # Bz GSM
        (123, 129),  # Proton density
        (129, 135),  # Bulk speed
        (218, 221),  # Kp index (I3) — stored as Kp*10, e.g. 3+ = 33
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
        "kp_raw",
        "dst",
    ]

    # OMNI2 uses multiple sentinel values across different fixed-width formats.
    # Explicit enumeration avoids misclassifying invalid measurements as
    # physical extremes.
    na_values = [
        "9999", " 9999", "999.9", " 999.9", "9999.", " 9999.",
        " 999.9", "999.99", "999999", " 999999", " 99999", "99999",
        "   99", "  9.9", "-999.9", " 999.99", "9999999", "999999.99",
        "99999", " 99999", "9999. ", " 9999. ",
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

        # Kp is stored as Kp*10 in the fixed-width format; divide to recover
        # the real value. Raw values ≥ 90 indicate missing data (fill value 99).
        df["kp"] = df["kp_raw"] / 10.0
        df.loc[df["kp"] > 9.0, "kp"] = float("nan")

        out = df[["datetime", "bt", "bz_gsm", "speed", "density", "dst", "kp"]].copy()
        out = out.sort_values("datetime").reset_index(drop=True)

        logger.info(
            "Parsed %s: %d rows, range %s → %s",
            filepath.name,
            len(out),
            out["datetime"].min(),
            out["datetime"].max(),
        )

        return out

    except Exception as err:
        logger.warning("Failed to parse %s: %s", filepath.name, err)
        return None


def parse_dscovr_json(data: list) -> Optional[pd.DataFrame]:
    """Parse a DSCOVR JSON feed returned by NOAA SWPC."""
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
