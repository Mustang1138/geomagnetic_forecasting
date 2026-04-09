"""Shared utility functions for the geomagnetic forecasting project."""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import yaml


def setup_logging(
        log_dir: str = "logs",
        level: int = logging.INFO,
) -> logging.Logger:
    """Configure project-wide logging to both file and console."""
    root_logger = logging.getLogger()

    # Idempotent: prevents duplicate handlers on repeated imports.
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
    module_logger.info("Logging initialised. Log file: %s", log_file)
    return module_logger


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load and validate project configuration from a YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as fh:
        cfg = yaml.safe_load(fh)

    try:
        data = cfg["data"]
        urls = data["urls"]

        if "dscovr" not in urls:
            raise KeyError("data.urls must define 'dscovr'")
        for key in ("mag", "plasma"):
            if key not in urls["dscovr"] or not urls["dscovr"][key]:
                raise KeyError(f"Missing required URL: data.urls.dscovr.{key}")

        if "omni2" not in urls:
            raise KeyError("data.urls must define 'omni2'")
        omni2 = urls["omni2"]
        if "base_url" not in omni2 or not omni2["base_url"]:
            raise KeyError("Missing required key: data.urls.omni2.base_url")
        if "filename_pattern" not in omni2 or not omni2["filename_pattern"]:
            raise KeyError("Missing required key: data.urls.omni2.filename_pattern")

        if "raw_dir" not in data or not data["raw_dir"]:
            raise KeyError("data.raw_dir must be set")
        if "processed_dir" not in data or not data["processed_dir"]:
            raise KeyError("data.processed_dir must be set")

    except KeyError as err:
        raise KeyError(f"Invalid configuration structure: {err}") from err

    return cfg


def load_pickle(path: Path) -> Any:
    """Deserialise a pickle file from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artefact not found: {path}. "
            "Ensure all preprocessing and training steps have been run."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it and any parents if necessary."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def find_project_root(start: Path, sentinel: str = "config.yaml") -> Path:
    """Walk up the directory tree from *start* until *sentinel* is found."""
    current = start.resolve()
    for _ in range(6):
        if (current / sentinel).exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        f"'{sentinel}' not found within six levels of '{start}'. "
        "Check the project structure."
    )


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Return True if *start_date* is strictly before *end_date*."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return start < end
    except ValueError:
        return False
