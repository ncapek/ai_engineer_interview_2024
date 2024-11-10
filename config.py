"""
This module provides configuration management utilities for the application.

It includes functionality for setting up logging, loading configurations from a YAML file,
and managing the paths to configuration files. The configuration file is expected to be
located in the 'config' directory, and the module will search for the file in the parent
directory of the script.

Modules:
    - setup_logging: Configures the basic logging setup for the application.
    - load_config: Loads configuration data from a YAML file and returns it as a dictionary.
"""

import logging
from pathlib import Path
from typing import Dict

import yaml


def setup_logging() -> None:
    """
    Set up basic logging configuration for the application.

    This function configures the root logger to log messages at the INFO level or higher.
    The log messages will include the timestamp, log level, and the message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from a YAML file located in the 'config' directory.

    Args:
        config_path (str): The relative path to the YAML configuration file to be loaded.
                            Defaults to 'config.yaml'.

    Returns:
        Dict: A dictionary representing the loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist in the expected path.
        yaml.YAMLError: If there is an error parsing the YAML file.
        Exception: If an unexpected error occurs while loading the configuration.
    """
    # Dynamically get the absolute path to the config.yaml file
    script_dir = Path(__file__).parent  # Get the directory where the script is located
    config_file_path = script_dir.parent / 'config' / config_path  # Navigate to config directory

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
