"""Migration v004: Per-tool circuit breaker state table."""

from __future__ import annotations

import asyncpg

from .migration_base import BaseMigration


class Migration(BaseMigration):
    version = 4
    name = "create_circuit_breaker_state"

    async def up(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                cb_id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tool_name               TEXT NOT NULL UNIQUE,
                status                  TEXT NOT NULL DEFAULT 'closed'
                                            CHECK (status IN ('closed', 'open', 'half_open')),
                failure_count           INTEGER NOT NULL DEFAULT 0,
                failure_threshold       INTEGER NOT NULL DEFAULT 5,
                opened_at               TIMESTAMPTZ,
                half_open_since         TIMESTAMPTZ,
                open_timeout_s          INTEGER NOT NULL DEFAULT 120,
                total_open_transitions  INTEGER NOT NULL DEFAULT 0,
                total_close_transitions INTEGER NOT NULL DEFAULT 0,
                last_failure_at         TIMESTAMPTZ,
                last_success_at         TIMESTAMPTZ,
                created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                version                 INTEGER NOT NULL DEFAULT 0
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cb_state_status
                ON circuit_breaker_state (status)
                WHERE status != 'closed'
        """)

    async def down(self, conn: asyncpg.Connection) -> None:
        await conn.execute("DROP TABLE IF EXISTS circuit_breaker_state CASCADE")
