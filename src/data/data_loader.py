"""
Data acquisition module for geomagnetic forecasting project.
"""

# NOTE:
# Although data_sources.py provides low-level HTTP access, this module
# intentionally re-implements acquisition logic in an orchestration context.
# This separation allows data_sources.py to remain stateless and reusable,
# while DataLoader manages experiment-specific workflows such as validation,
# persistence, and dataset assembly (Fielding, 2000; Martin, 2008).

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.evaluation.validators import validate_omni_dataframe
from src.preprocessing.parsers import parse_omni2_file, parse_dscovr_json
from src.utils import load_config, ensure_dir, setup_logging

logger = setup_logging()


class DataLoader:
    """
    Orchestrates data acquisition, parsing, validation, and persistence.

    Separating orchestration from parsing logic improves maintainability
    and supports systematic experimentation (Martin, 2008).
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        self.raw_dir = Path(self.config["data"]["raw_dir"])
        ensure_dir(self.raw_dir)

        self.urls = self.config["data"]["urls"]

        logger.info("DataLoader initialised")

    # OMNI2 Historical Data

    def download_omni2_year(self, year: int) -> bool:
        omni_cfg = self.urls["omni2"]

        filename = omni_cfg["filename_pattern"].format(year=year)
        url = omni_cfg["base_url"] + filename
        out_path = self.raw_dir / filename

        if out_path.exists():
            logger.info(f"✓ OMNI2 {year} already exists")
            return True

        try:
            logger.info(f"Downloading OMNI2 {year}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except requests.RequestException as e:
            logger.error(f"Failed OMNI2 {year}: {e}")
            return False

    # Network or availability failures are logged rather than raised
    # to allow long-range historical downloads to complete partially.
    # This design prioritises robustness over strict completeness,
    # which is appropriate for large-scale scientific datasets
    # (Lwakatare et al., 2020).

    def download_omni2_range(self, start_year: int, end_year: int) -> None:
        for year in range(start_year, end_year + 1):
            self.download_omni2_year(year)
            time.sleep(0.5)

    def load_omni2_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Load and combine OMNI2 data across multiple years.

        Chronological ordering and validation are essential to prevent
        temporal leakage and ensure suitability for time series models
        (Cerqueira et al., 2020).
        """
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

    # Fetch DSCOVR Real-time Data

    def fetch_dscovr(self, kind: str) -> Optional[pd.DataFrame]:
        """
        Fetch real-time DSCOVR data.

        This data is excluded from training to avoid inconsistencies
        between historical and operational data streams
        (Cristoforetti et al., 2022).
        """
        url = self.urls["dscovr"][kind]

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return parse_dscovr_json(r.json())
        except requests.RequestException:
            return None

    # Save as CSV

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        if df.empty:
            logger.warning(f"Empty DataFrame not saved: {filename}")
            return

        path = self.raw_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows → {path}")


def main():
    loader = DataLoader()

    cfg = loader.config["data"]["date_range"]
    start_year = int(cfg["start"][:4])
    end_year = int(cfg["end"][:4])

    loader.download_omni2_range(start_year, end_year)
    omni_df = loader.load_omni2_range(start_year, end_year)
    loader.save_csv(omni_df, "omni2_combined.csv")

    mag = loader.fetch_dscovr("mag")
    plasma = loader.fetch_dscovr("plasma")

    if mag is not None:
        loader.save_csv(mag, "dscovr_mag_realtime.csv")
    if plasma is not None:
        loader.save_csv(plasma, "dscovr_plasma_realtime.csv")


if __name__ == "__main__":
    main()
