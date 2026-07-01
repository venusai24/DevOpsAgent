"""Migration v003: Baseline Statistics Store (partitioned PostgreSQL tables).

Spec: StateManagement&PersistenceLayer.md §3.4
Partitioned by ingestion week via range partitioning on computed_at.
"""

from __future__ import annotations

import asyncpg

from .migration_base import BaseMigration


class Migration(BaseMigration):
    version = 3
    name = "create_baseline_store"

    async def up(self, conn: asyncpg.Connection) -> None:
        # Aggregate statistics table (one row per baseline × component × KPI × granularity).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS baseline_statistics (
                id                  BIGSERIAL,
                baseline_ref        TEXT NOT NULL,
                cmdb_id             TEXT NOT NULL,
                kpi_name            TEXT NOT NULL,
                mean                DOUBLE PRECISION NOT NULL,
                stddev              DOUBLE PRECISION NOT NULL,
                p50                 DOUBLE PRECISION NOT NULL,
                p95                 DOUBLE PRECISION NOT NULL,
                p99                 DOUBLE PRECISION NOT NULL,
                sample_count        INTEGER NOT NULL,
                granularity_minutes INTEGER NOT NULL DEFAULT 1,
                computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (baseline_ref, cmdb_id, kpi_name, granularity_minutes)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_baseline_stat_ref
                ON baseline_statistics (baseline_ref)
        """)
        # Time-bucket table (one row per baseline × component × KPI × bucket_start).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS baseline_time_buckets (
                id            BIGSERIAL,
                baseline_ref  TEXT NOT NULL,
                cmdb_id       TEXT NOT NULL,
                kpi_name      TEXT NOT NULL,
                bucket_start  TIMESTAMPTZ NOT NULL,
                mean          DOUBLE PRECISION NOT NULL,
                stddev        DOUBLE PRECISION NOT NULL,
                p50           DOUBLE PRECISION NOT NULL,
                p95           DOUBLE PRECISION NOT NULL,
                p99           DOUBLE PRECISION NOT NULL,
                sample_count  INTEGER NOT NULL,
                PRIMARY KEY (baseline_ref, cmdb_id, kpi_name, bucket_start)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_baseline_bucket_ref_time
                ON baseline_time_buckets (baseline_ref, bucket_start)
        """)

    async def down(self, conn: asyncpg.Connection) -> None:
        await conn.execute("DROP TABLE IF EXISTS baseline_time_buckets CASCADE")
        await conn.execute("DROP TABLE IF EXISTS baseline_statistics CASCADE")
