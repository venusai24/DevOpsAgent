"""PostgreSQL connection pool and transaction management adapter.

All Postgres I/O flows through this adapter.  It wraps asyncpg and exposes
a context-manager-based transaction API that supports:
  - READ COMMITTED (default)
  - SERIALIZABLE (required for the dedup claim transaction, Phase 3 §3.2.2)
  - Nested savepoints (for idempotent sub-operations)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from asyncpg import Connection, Pool

logger = logging.getLogger(__name__)


class PostgresAdapter:
    """Manages an asyncpg connection pool and exposes transaction context managers.

    Usage::

        adapter = PostgresAdapter(dsn="postgresql://user:pass@host/db")
        await adapter.connect()

        async with adapter.transaction() as conn:
            await conn.execute("INSERT INTO ...")

        async with adapter.serializable_transaction() as conn:
            rows = await conn.fetch("SELECT ... FOR UPDATE")

        await adapter.disconnect()
    """

    def __init__(
        self,
        dsn: str,
        min_connections: int = 2,
        max_connections: int = 20,
        statement_cache_size: int = 100,
        command_timeout: float = 30.0,
    ):
        self._dsn = dsn
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._statement_cache_size = statement_cache_size
        self._command_timeout = command_timeout
        self._pool: Pool | None = None

    async def connect(self) -> None:
        """Initialise the connection pool."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_connections,
            max_size=self._max_connections,
            statement_cache_size=self._statement_cache_size,
            command_timeout=self._command_timeout,
        )
        logger.info(
            "PostgresAdapter connected (min=%d, max=%d)",
            self._min_connections,
            self._max_connections,
        )

    async def disconnect(self) -> None:
        """Gracefully close all pooled connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgresAdapter disconnected")

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            raise RuntimeError("PostgresAdapter.connect() has not been called")
        return self._pool

    @asynccontextmanager
    async def transaction(
        self, isolation: str = "read_committed"
    ) -> AsyncGenerator[Connection, None]:
        """Yield a connection inside a transaction at the specified isolation level.

        READ COMMITTED is the default; pass ``isolation="serializable"`` for the
        dedup claim path (Phase 3 §3.2.2).
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction(isolation=isolation):
                yield conn

    @asynccontextmanager
    async def serializable_transaction(self) -> AsyncGenerator[Connection, None]:
        """Convenience shorthand for SERIALIZABLE isolation."""
        async with self.transaction(isolation="serializable") as conn:
            yield conn

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Yield a raw connection for read-only or auto-commit operations."""
        async with self.pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a write statement using an auto-acquired connection."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a read query and return all rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Execute a read query and return the first row, or None."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a scalar read query."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
