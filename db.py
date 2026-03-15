"""
Database connection management for ScholarRAG.

Provides a lightweight connection pool (psycopg_pool) used by the indexer
and dense retriever. Import `get_pool()` wherever a DB connection is needed.

Usage:
    from db import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            print(cur.fetchone())
"""

from __future__ import annotations

from functools import lru_cache

import psycopg
import psycopg_pool
from pgvector.psycopg import register_vector

from config import settings
from logger import get_logger

log = get_logger(__name__)


def _configure_conn(conn: psycopg.Connection) -> None:
    """Called for every new connection in the pool."""
    register_vector(conn)


@lru_cache(maxsize=1)
def get_pool() -> psycopg_pool.ConnectionPool:
    """
    Return the shared connection pool. Created once per process.
    Min 1 connection kept warm, max 10 for concurrent API requests.
    """
    log.info("Opening connection pool → %s", settings.database_url)
    pool = psycopg_pool.ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure_conn,
        open=True,
    )
    pool.wait()  # Block until at least one connection is established
    log.info("Connection pool ready.")
    return pool


def close_pool() -> None:
    """Gracefully close the pool on shutdown (call from FastAPI lifespan)."""
    pool = get_pool()
    pool.close()
    log.info("Connection pool closed.")


def ping() -> bool:
    """Return True if the database is reachable. Safe to call at startup."""
    try:
        pool = get_pool()
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:
        log.error("Database ping failed: %s", exc)
        return False
