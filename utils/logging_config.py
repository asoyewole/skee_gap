"""Centralized logging configuration for the project.

Provides a get_logger(name) helper so modules can get a properly configured logger.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from typing import Optional


LOG_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skee_gap.log")


def _configure_root_logger(level: int = logging.DEBUG, logfile: Optional[str] = LOG_FILENAME) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # try to add file handler where possible
    try:
        fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        # allow running without file logging
        root.debug("Could not create log file handler; continuing without file logging")


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    _configure_root_logger(level=level)
    return logging.getLogger(name)
