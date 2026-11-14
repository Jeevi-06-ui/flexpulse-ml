"""
Structured logging module for FlexPulse-ML.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logger(
    name: str = "flexpulse",
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = log_format or "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        dfmt = date_format or "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=dfmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "flexpulse") -> logging.Logger:
    return logging.getLogger(name)
