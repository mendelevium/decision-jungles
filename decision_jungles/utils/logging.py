"""
Logging utilities for Decision Jungles.

This module provides logging functionality for the Decision Jungles library,
with different verbosity levels and consistent formatting.
"""

import logging
import sys
from typing import Optional, Union, TextIO

# Define logger levels with descriptive names
NOTSET = logging.NOTSET  # 0
DEBUG = logging.DEBUG    # 10
INFO = logging.INFO      # 20
WARNING = logging.WARNING  # 30
ERROR = logging.ERROR    # 40
CRITICAL = logging.CRITICAL  # 50

# Create a logger for the decision_jungles package
logger = logging.getLogger("decision_jungles")


def setup_logging(
    level: int = INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """
    Configure the logging system for Decision Jungles.
    
    Parameters
    ----------
    level : int, default=INFO
        The logging level to use. Options include:
        - NOTSET (0): All messages
        - DEBUG (10): Debug messages and above
        - INFO (20): Informational messages and above
        - WARNING (30): Warning messages and above
        - ERROR (40): Error messages and above
        - CRITICAL (50): Critical error messages only
    
    format_string : str, optional
        The format string to use for log messages. If None, a default format is used.
    
    log_file : str, optional
        If provided, log output will also be written to this file.
    
    stream : TextIO, optional
        The stream to use for logging (default: sys.stderr).
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Set the log level
    logger.setLevel(level)
    
    # Add a stream handler (console)
    stream_handler = logging.StreamHandler(stream or sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Optionally add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Parameters
    ----------
    name : str
        The name of the module, typically __name__.
    
    Returns
    -------
    logging.Logger
        A logger instance for the specified module.
    """
    return logging.getLogger(f"decision_jungles.{name}")


# Set up default logging configuration
setup_logging()