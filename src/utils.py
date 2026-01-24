"""
Utility functions for the geomagnetic forecasting project.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

import yaml


def setup_logging(
        log_dir: str = "logs",
        level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the project.

    Centralised logging supports reproducibility, debugging, and
    post-hoc analysis of long-running machine learning experiments
    (Lwakatare et al., 2020).
    """
    root_logger = logging.getLogger()

    # Idempotent check prevents duplicate handlers when modules
    # are imported multiple times, a common issue in ML pipelines
    # (Martin, 2008).
    if root_logger.handlers:
        return logging.getLogger(__name__)

    Path(log_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"geomag_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    module_logger = logging.getLogger(__name__)
    module_logger.info(f"Logging initialised. Log file: {log_file}")
    return module_logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate project configuration from a YAML file.

    Early structural validation reduces runtime failure risk and
    enforces explicit assumptions about data sources and directories,
    aligning with defensive programming principles (Martin, 2008).
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Validation
    try:
        data = cfg["data"]
        urls = data["urls"]

        # Validate DSCOVR URLs
        if "dscovr" not in urls:
            raise KeyError("data.urls must define 'dscovr'")
        for key in ("mag", "plasma"):
            if key not in urls["dscovr"] or not urls["dscovr"][key]:
                raise KeyError(f"Missing required URL: data.urls.dscovr.{key}")

        # Validate OMNI2 URLs
        if "omni2" not in urls:
            raise KeyError("data.urls must define 'omni2'")
        omni2 = urls["omni2"]
        if "base_url" not in omni2 or not omni2["base_url"]:
            raise KeyError(
                "Missing required OMNI2 base_url: data.urls.omni2.base_url")
        if "filename_pattern" not in omni2 or not omni2["filename_pattern"]:
            raise KeyError(
                "Missing required OMNI2 filename_pattern: data.urls.omni2.filename_pattern")

        # Validate directories
        if "raw_dir" not in data or not data["raw_dir"]:
            raise KeyError("data.raw_dir must be set")
        if "processed_dir" not in data or not data["processed_dir"]:
            raise KeyError("data.processed_dir must be set")

    except KeyError as err:
        raise KeyError(f"Invalid configuration structure: {err}")

    return cfg


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate that the date range is logically ordered.

    Ensuring correct temporal ordering is essential for time series
    modelling, particularly for sequence-based models such as LSTMs
    (Box et al., 2015).
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return start < end
    except ValueError:
        return False
