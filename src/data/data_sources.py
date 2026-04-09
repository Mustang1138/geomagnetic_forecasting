"""HTTP client functions for fetching raw OMNI2 and DSCOVR data."""

import logging
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.utils import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_config = load_config()
_urls = _config["data"]["urls"]

DSCOVR_MAG_URL = _urls["dscovr"]["mag"]
DSCOVR_PLASMA_URL = _urls["dscovr"]["plasma"]
OMNI2_BASE_URL = _urls["omni2"]["base_url"]


def fetch_dscovr_mag() -> list[Any]:
    """Fetch raw DSCOVR magnetic field data in JSON format."""
    logger.info("Fetching DSCOVR magnetic field data")
    response = requests.get(DSCOVR_MAG_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_dscovr_plasma() -> list[Any]:
    """Fetch raw DSCOVR plasma data in JSON format."""
    logger.info("Fetching DSCOVR plasma data")
    response = requests.get(DSCOVR_PLASMA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def download_omni2_year(year: int, output_dir: Path) -> None:
    """Download a single OMNI2 annual data file, skipping if already present."""
    omni2_cfg = _urls["omni2"]
    filename = omni2_cfg["filename_pattern"].format(year=year)
    url = omni2_cfg["base_url"] + filename
    output_file = output_dir / filename

    if output_file.exists():
        logger.info("OMNI2 %d already exists — skipping.", year)
        return

    logger.info("Downloading OMNI2 data for %d.", year)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_file.write_bytes(response.content)
    logger.info("Downloaded OMNI2 %d (%.1f KB).", year, len(response.content) / 1024)

    time.sleep(0.5)


def download_omni2_range(start_year: int, end_year: int, output_dir: Path) -> None:
    """Download OMNI2 data for an inclusive range of years."""
    logger.info("Downloading OMNI2 data: %d–%d", start_year, end_year)

    for year in tqdm(range(start_year, end_year + 1), desc="Downloading OMNI2"):
        try:
            download_omni2_year(year, output_dir)
        except requests.RequestException as exc:
            logger.error("Failed to download OMNI2 %d: %s", year, exc)
