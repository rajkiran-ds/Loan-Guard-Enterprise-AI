"""
src/utils/logger.py
--------------------
Central logging factory. Every module calls `get_logger(__name__)` instead of
configuring its own handlers, so log formatting stays consistent app-wide.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "loanguard.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_configured_loggers: set[str] = set()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a module-scoped logger with console + rotating-file handlers."""
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger
