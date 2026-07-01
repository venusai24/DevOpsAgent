"""MigrationManager — applies and tracks schema migrations.

Each migration is an idempotent class with up() and down() methods.
Applied migrations are tracked in the schema_migrations table.
"""

from __future__ import annotations

import importlib
import logging

from ..adapters.postgres_adapter import PostgresAdapter

logger = logging.getLogger(__name__)

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version      INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    checksum     TEXT
);
"""


class MigrationManager:
    """Applies schema migrations in order and tracks applied versions.

    Usage::

        mgr = MigrationManager(adapter)
        await mgr.migrate()        # apply all pending migrations
        await mgr.migrate(target=3)  # migrate to version 3 only
        await mgr.rollback(2)      # roll back to version 2
    """

    def __init__(self, adapter: PostgresAdapter, migration_modules: list[str] | None = None):
        self._db = adapter
        # Default migration module paths in version order.
        self._migration_modules = migration_modules or [
            "devops_agent.state.migrations.v001_investigation_registry",
            "devops_agent.state.migrations.v002_checkpoint_tables",
            "devops_agent.state.migrations.v003_baseline_store",
            "devops_agent.state.migrations.v004_circuit_breaker_state",
        ]

    async def ensure_migrations_table(self) -> None:
        await self._db.execute(_MIGRATIONS_TABLE)

    async def applied_versions(self) -> list[int]:
        rows = await self._db.fetch(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        return [r["version"] for r in rows]

    async def migrate(self, target: int | None = None) -> list[int]:
        """Apply all pending migrations up to (and including) ``target``.

        Returns the list of newly applied versions.
        """
        await self.ensure_migrations_table()
        applied = set(await self.applied_versions())
        newly_applied = []

        for module_path in self._migration_modules:
            module = importlib.import_module(module_path)
            migration = module.Migration()
            if migration.version in applied:
                continue
            if target is not None and migration.version > target:
                break
            logger.info("Applying migration v%03d: %s", migration.version, migration.name)
            async with self._db.transaction() as conn:
                await migration.up(conn)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
                    migration.version, migration.name,
                )
            newly_applied.append(migration.version)
            logger.info("Migration v%03d applied", migration.version)

        return newly_applied

    async def rollback(self, to_version: int) -> list[int]:
        """Roll back all migrations newer than ``to_version``.

        Returns the list of rolled-back versions.
        """
        await self.ensure_migrations_table()
        applied = sorted(await self.applied_versions(), reverse=True)
        rolled_back = []

        for module_path in reversed(self._migration_modules):
            module = importlib.import_module(module_path)
            migration = module.Migration()
            if migration.version not in applied:
                continue
            if migration.version <= to_version:
                break
            logger.info("Rolling back migration v%03d: %s", migration.version, migration.name)
            async with self._db.transaction() as conn:
                await migration.down(conn)
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE version = $1", migration.version
                )
            rolled_back.append(migration.version)
            logger.info("Migration v%03d rolled back", migration.version)

        return rolled_back
