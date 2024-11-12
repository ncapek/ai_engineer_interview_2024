"""
This module provides configuration management utilities for the application.
"""

import logging
from pathlib import Path
from typing import Dict

import yaml

def load_config(config_path: str = "config.yml") -> Dict:
    """
    Load configuration from a YAML file located in the 'config' directory.
    """
    config_file_path = Path(config_path)  # Navigate to config directory

    # Check if the file exists before trying to open it
    if not config_file_path.exists():
        logging.error("Configuration file '%s' not found.", config_file_path)
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

    try:
        with open(config_file_path, "r", encoding="utf-8") as file:  # Specify encoding
            config = yaml.safe_load(file)
            logging.info("Configuration loaded successfully from %s", config_file_path)
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML file: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise

    return config
