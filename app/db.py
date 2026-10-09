from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

_pool: ConnectionPool | None = None


def init_db() -> None:
    """Create extension, table and indexes (safe to run repeatedly)."""
    schema = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(schema)


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)  # teach psycopg about the vector type
    try:
        # pgvector 0.8+: keep scanning the index until enough rows pass the filter
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except psycopg.Error:
        conn.rollback()


def open_pool() -> ConnectionPool:
    global _pool
    _pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
        configure=_configure,
        open=True,
    )
    return _pool


def close_pool() -> None:
    if _pool:
        _pool.close()


def get_pool() -> ConnectionPool:
    assert _pool is not None, "Pool not opened"
    return _pool