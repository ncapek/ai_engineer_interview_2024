"""
utils.py

This module contains utility functions for the logging setup in the application.
The logger is set up with a specific logging level and format.
"""

import logging


def setup_logger(name: str, log_file: str = 'app.log', level=logging.DEBUG):
    """
    Set up a logger that writes log messages to both a file and the console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream handler to print logs to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger
