"""
Logging configuration for ScholarRAG.
Call `configure_logging()` once at the entry point of each process
(API server, CLI scripts, eval harness).
"""

import logging
import sys

from config import settings


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use `__name__` as the argument."""
    return logging.getLogger(name)
