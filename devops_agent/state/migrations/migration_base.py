"""Abstract base class for all database migrations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import asyncpg


class BaseMigration(ABC):
    """Contract for all schema migrations.

    - ``version`` must be a unique monotonically increasing integer.
    - ``name`` is a human-readable description.
    - ``up()`` is idempotent (safe to re-run).
    - ``down()`` reverses the migration exactly.
    """

    @property
    @abstractmethod
    def version(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def up(self, conn: asyncpg.Connection) -> None:
        """Apply the migration."""

    @abstractmethod
    async def down(self, conn: asyncpg.Connection) -> None:
        """Reverse the migration."""
