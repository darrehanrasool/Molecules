"""Standardized logging utility."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "molecules", log_file: Optional[str] = None) -> logging.Logger:
    """Returns a configured logger instance with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
