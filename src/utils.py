"""
Utility functions for the geomagnetic forecasting project.

This module provides common helper functions for:
- Configuration loading
- Logging setup
- Directory management
- Data validation
"""

import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the project.

    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"geomag_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Path to directory

    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate that date range is logical.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        True if valid, False otherwise
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return start < end
    except ValueError:
        return False


if __name__ == "__main__":
    """Test utility functions"""

    print("=" * 60)
    print("Testing Geomagnetic Forecasting Utilities")
    print("=" * 60)

    # Test config loading
    print("\n1. Testing configuration loading...")
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  - Data sources: {list(config['data']['sources'].keys())}")
        print(f"  - Date range: {config['data']['date_range']['start']} to {config['data']['date_range']['end']}")
        print(f"  - Models: {list(config['models'].keys())}")
    except Exception as e:
        print(f"✗ Error loading config: {e}")

    # Test directory creation
    print("\n2. Testing directory management...")
    test_dir = ensure_dir("test_output")
    print(f"✓ Directory ensured: {test_dir}")
    test_dir.rmdir()  # Clean up

    # Test date validation
    print("\n3. Testing date validation...")
    valid = validate_date_range("2010-01-01", "2024-12-31")
    print(f"✓ Date range valid: {valid}")

    # Test logging
    print("\n4. Testing logging setup...")
    logger = setup_logging()
    logger.info("This is a test log message")
    print("✓ Logging configured")

    print("\n" + "=" * 60)
    print("All utility tests completed!")
    print("=" * 60)