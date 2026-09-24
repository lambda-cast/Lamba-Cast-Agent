"""PostgreSQL connection pool for the solar/weather database.

Config is read from environment variables (see .env):

  POSTGRES_DSN            full DSN, e.g. "postgresql://user:pass@host:5432/db"
                           (takes precedence if set)
  PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
                           used to build a DSN if POSTGRES_DSN is not set
"""

import os
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_DSN = os.getenv("POSTGRES_DSN") or (
    f"host={os.getenv('PG_HOST', 'localhost')} "
    f"port={os.getenv('PG_PORT', '5432')} "
    f"dbname={os.getenv('PG_DATABASE', '')} "
    f"user={os.getenv('PG_USER', '')} "
    f"password={os.getenv('PG_PASSWORD', '')}"
)

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Lazily create a singleton connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_DSN,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def get_cursor():
    """Context manager yielding a dict-row cursor from the pool."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            yield cur