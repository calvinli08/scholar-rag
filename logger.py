"""
Logging configuration for ScholarRAG.
Call `configure_logging()` once at the entry point of each process
(API server, CLI scripts, eval harness).
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import settings

# Project root directory (parent of this file)
PROJECT_ROOT = Path(__file__).resolve().parent
ERROR_LOG_PATH = PROJECT_ROOT / "error.log"


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create root logger and set level
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(console_handler)

    # File handler for error.log
    file_handler = RotatingFileHandler(
        filename=ERROR_LOG_PATH,
        mode="a",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)
    
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use `__name__` as the argument."""
    return logging.getLogger(name)
