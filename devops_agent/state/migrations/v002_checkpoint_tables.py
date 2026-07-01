"""Migration v002: checkpoints table for the LangGraph data-plane store."""

from __future__ import annotations

import asyncpg

from .migration_base import BaseMigration


class Migration(BaseMigration):
    version = 2
    name = "create_checkpoints_table"

    async def up(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id           UUID PRIMARY KEY,
                thread_id               UUID NOT NULL,
                parent_checkpoint_id    UUID,
                producing_node          TEXT NOT NULL DEFAULT '',
                sequence_number         INTEGER NOT NULL DEFAULT 0,
                created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                payload                 JSONB,
                payload_schema_version  INTEGER NOT NULL DEFAULT 1,
                is_latest               BOOLEAN NOT NULL DEFAULT TRUE,
                checkpoint_status       TEXT NOT NULL DEFAULT 'active',
                channel_versions        JSONB NOT NULL DEFAULT '{}',
                pending_writes          JSONB NOT NULL DEFAULT '[]',
                version                 INTEGER NOT NULL DEFAULT 0
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_latest
                ON checkpoints (thread_id, is_latest)
                WHERE is_latest = TRUE
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_seq
                ON checkpoints (thread_id, sequence_number DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_parent
                ON checkpoints (parent_checkpoint_id)
                WHERE parent_checkpoint_id IS NOT NULL
        """)

    async def down(self, conn: asyncpg.Connection) -> None:
        await conn.execute("DROP TABLE IF EXISTS checkpoints CASCADE")
