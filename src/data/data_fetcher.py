"""Data acquisition module: downloads, parses, validates, and saves OMNI2 and DSCOVR data."""

import calendar
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.evaluation.validators import validate_omni_dataframe
from src.preprocessing.parsers import parse_dscovr_json, parse_omni2_file
from src.utils import ensure_dir, load_config, setup_logging

logger = setup_logging()


def resolve_end_date(end_value: str) -> datetime:
    """Resolve the configured end date to a concrete datetime object.

    If ``end_value`` is ``"auto"``, returns the last day of the previous
    calendar month. Otherwise parses the value as ``YYYY-MM-DD``.
    """
    if end_value.strip().lower() == "auto":
        today = datetime.now()

        if today.month == 1:
            year, month = today.year - 1, 12
        else:
            year, month = today.year, today.month - 1

        last_day = calendar.monthrange(year, month)[1]
        return datetime(year, month, last_day)

    try:
        return datetime.strptime(end_value.strip(), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid end date '{end_value}'. "
            "Expected 'auto' or a date in YYYY-MM-DD format."
        ) from exc


class DataLoader:
    """Orchestrates data acquisition, parsing, validation, and persistence."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        self.raw_dir = Path(self.config["data"]["raw_dir"])
        ensure_dir(self.raw_dir)

        self.urls = self.config["data"]["urls"]

        logger.info("DataLoader initialised.")

    def download_omni2_year(self, year: int) -> bool:
        """Download a single year of OMNI2 data, skipping if already on disk."""
        omni_cfg = self.urls["omni2"]
        filename = omni_cfg["filename_pattern"].format(year=year)
        url = omni_cfg["base_url"] + filename
        out_path = self.raw_dir / filename

        if out_path.exists():
            logger.info("OMNI2 %d already exists — skipping.", year)
            return True

        try:
            logger.info("Downloading OMNI2 %d …", year)
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            return True
        except requests.RequestException as exc:
            # Log rather than raise so that a long-range historical download
            # can complete partially despite transient network failures.
            logger.error("Failed to download OMNI2 %d: %s", year, exc)
            return False

    def download_omni2_range(self, start_year: int, end_year: int) -> None:
        """Download OMNI2 data for an inclusive range of years."""
        for year in range(start_year, end_year + 1):
            self.download_omni2_year(year)
            time.sleep(0.5)

    def load_omni2_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load and combine OMNI2 data across multiple years into a single sorted DataFrame."""
        frames: list[pd.DataFrame] = []

        for year in range(start_year, end_year + 1):
            path = self.raw_dir / f"omni2_{year}.dat"
            if not path.exists():
                continue

            df = parse_omni2_file(path)
            if df is not None:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        validate_omni_dataframe(combined)
        return combined

    def fetch_dscovr(self, kind: str) -> Optional[pd.DataFrame]:
        """Fetch real-time DSCOVR solar wind data from NOAA SWPC."""
        url = self.urls["dscovr"][kind]

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return parse_dscovr_json(response.json())
        except requests.RequestException:
            return None

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame to CSV in the raw data directory."""
        if df.empty:
            logger.warning("Empty DataFrame — skipping save: %s", filename)
            return

        path = self.raw_dir / filename
        df.to_csv(path, index=False)
        logger.info("Saved %d rows → %s", len(df), path)


def main() -> None:
    """Entry point for the data acquisition pipeline."""
    loader = DataLoader()

    date_range_cfg = loader.config["data"]["date_range"]

    start_year = int(date_range_cfg["start"][:4])
    end_date = resolve_end_date(date_range_cfg["end"])
    end_year = end_date.year

    logger.info(
        "Data range: %d → %d (%s)",
        start_year,
        end_year,
        end_date.strftime("%Y-%m-%d"),
    )

    loader.download_omni2_range(start_year, end_year)
    omni_df = loader.load_omni2_range(start_year, end_year)
    loader.save_csv(omni_df, "omni2_combined.csv")

    mag_df = loader.fetch_dscovr("mag")
    plasma_df = loader.fetch_dscovr("plasma")

    if mag_df is not None:
        loader.save_csv(mag_df, "dscovr_mag_realtime.csv")
    if plasma_df is not None:
        loader.save_csv(plasma_df, "dscovr_plasma_realtime.csv")


if __name__ == "__main__":
    main()
