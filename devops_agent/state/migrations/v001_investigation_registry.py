"""Migration v001: investigation_registry table and indexes.

Spec: StateManagement&PersistenceLayer.md §3.1.2
"""

from __future__ import annotations

import asyncpg

from .migration_base import BaseMigration


class Migration(BaseMigration):
    version = 1
    name = "create_investigation_registry"

    async def up(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TYPE IF NOT EXISTS investigation_status AS ENUM (
                'claimed', 'pending_context', 'active', 'hitl_paused',
                'resolved', 'superseded', 'duplicate_halted',
                'agent_failure', 'timeout'
            )
        """)
        await conn.execute("""
            CREATE TYPE IF NOT EXISTS dedup_decision_type AS ENUM (
                'NEW', 'DUPLICATE', 'SUBSET', 'SUPERSET', 'PARTIAL_OVERLAP'
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS investigation_registry (
                investigation_id              UUID PRIMARY KEY,
                cluster_id                    TEXT NOT NULL,
                time_range                    TSTZRANGE NOT NULL,
                symptom_signature             TEXT NOT NULL,
                affected_component_candidates TEXT[] NOT NULL DEFAULT '{}',
                status                        investigation_status NOT NULL DEFAULT 'claimed',
                dedup_decision                dedup_decision_type,
                related_investigation_id      UUID REFERENCES investigation_registry(investigation_id),
                topology_manifest_version     TEXT,
                stage0_cache_key              TEXT,
                confidence_level              TEXT,
                root_cause_cmdb_id            TEXT,
                current_node                  TEXT,
                created_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at                  TIMESTAMPTZ,
                version                       INTEGER NOT NULL DEFAULT 0,
                CONSTRAINT chk_related_not_self
                    CHECK (related_investigation_id IS DISTINCT FROM investigation_id)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inv_registry_time_range
                ON investigation_registry USING gist (time_range)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inv_registry_components
                ON investigation_registry USING gin (affected_component_candidates)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inv_registry_live_by_cluster
                ON investigation_registry (cluster_id, status)
                WHERE status NOT IN ('resolved', 'superseded', 'duplicate_halted',
                                     'agent_failure', 'timeout')
        """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_inv_registry_live_duplicate
                ON investigation_registry (cluster_id, symptom_signature)
                WHERE status NOT IN ('resolved', 'superseded', 'duplicate_halted',
                                     'agent_failure', 'timeout')
        """)

    async def down(self, conn: asyncpg.Connection) -> None:
        await conn.execute("DROP TABLE IF EXISTS investigation_registry CASCADE")
        await conn.execute("DROP TYPE IF EXISTS investigation_status CASCADE")
        await conn.execute("DROP TYPE IF EXISTS dedup_decision_type CASCADE")
