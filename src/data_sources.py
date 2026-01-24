"""
Data source clients for geomagnetic forecasting project.

Handles all external data access (HTTP requests, file downloads).

This module intentionally performs no parsing or transformation,
following a layered architecture approach that separates data access
from data interpretation (Fielding, 2000).
"""

import logging
import time
from pathlib import Path
from typing import Any, List

import requests
from tqdm import tqdm

from src.utils import load_config
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_config = load_config()
_urls = _config["data"]["urls"]

DSCOVR_MAG_URL = _urls["dscovr_mag"]
DSCOVR_PLASMA_URL = _urls["dscovr_plasma"]
OMNI2_BASE_URL = _urls["omni2_base"]


def fetch_dscovr_mag() -> List[Any]:
    """
    Fetch raw DSCOVR magnetic field data in JSON format.

    Real-time DSCOVR data is useful for exploratory analysis but is
    not used for model training due to higher noise and missing data
    rates (NOAA SWPC, 2024; Cristoforetti et al., 2022).
    """
    logger.info("Fetching DSCOVR magnetic field data")
    response = requests.get(DSCOVR_MAG_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_dscovr_plasma() -> List[Any]:
    """Fetch raw DSCOVR plasma JSON."""
    logger.info("Fetching DSCOVR plasma data")
    response = requests.get(DSCOVR_PLASMA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def download_omni2_year(year: int, output_dir: Path) -> None:
    """
    Download a single OMNI2 annual data file.

    OMNI2 is a standard benchmark dataset for geomagnetic research
    and provides hourly-averaged solar wind and geomagnetic indices
    (Papitashvili and King, 2020).
    """
    url = f"{OMNI2_BASE_URL}{year}.dat"
    output_file = output_dir / f"omni2_{year}.dat"

    if output_file.exists():
        logger.info(f"✓ OMNI2 {year} already exists, skipping")
        return

    logger.info(f"Downloading OMNI2 data for {year}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_file.write_bytes(response.content)
    logger.info(f"✓ Downloaded {year} ({len(response.content) / 1024:.1f} KB)")

    time.sleep(0.5)


def download_omni2_range(start_year: int, end_year: int, output_dir: Path) -> None:
    """Download OMNI2 data for a year range."""
    logger.info(f"Downloading OMNI2 data: {start_year}–{end_year}")

    for year in tqdm(range(start_year, end_year + 1), desc="Downloading OMNI2"):
        try:
            download_omni2_year(year, output_dir)
        except requests.RequestException as e:
            logger.error(f"✗ Failed to download {year}: {e}")
