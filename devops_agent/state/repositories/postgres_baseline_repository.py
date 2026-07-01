"""Concrete BaselineRepository backed by partitioned PostgreSQL tables."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from ..adapters.postgres_adapter import PostgresAdapter
from ..interfaces.baseline_repository import (
    BaselineRepository,
    BaselineStatistics,
    BaselineTimeBucket,
)

logger = logging.getLogger(__name__)


class PostgresBaselineRepository(BaselineRepository):
    """Partitioned by ingestion week (baseline_ref prefix encodes the week).

    See Phase 3 §3.4.2 for the recommended partition strategy.
    """

    def __init__(self, adapter: PostgresAdapter):
        self._db = adapter

    async def write_statistics(
        self,
        baseline_ref: str,
        rows: Sequence[BaselineStatistics],
    ) -> str:
        """Batch-insert statistics rows; idempotent via ON CONFLICT DO NOTHING."""
        if not rows:
            return baseline_ref
        async with self._db.transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO baseline_statistics (
                    baseline_ref, cmdb_id, kpi_name, mean, stddev,
                    p50, p95, p99, sample_count, granularity_minutes, computed_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (baseline_ref, cmdb_id, kpi_name, granularity_minutes)
                DO NOTHING
                """,
                [
                    (
                        r.baseline_ref, r.cmdb_id, r.kpi_name,
                        r.mean, r.stddev, r.p50, r.p95, r.p99,
                        r.sample_count, r.granularity_minutes,
                        r.computed_at or datetime.utcnow(),
                    )
                    for r in rows
                ],
            )
        return baseline_ref

    async def get_statistics(
        self,
        baseline_ref: str,
        cmdb_id: str,
        kpi_name: str,
    ) -> BaselineStatistics | None:
        row = await self._db.fetchrow(
            """
            SELECT baseline_ref, cmdb_id, kpi_name, mean, stddev,
                   p50, p95, p99, sample_count, granularity_minutes, computed_at
            FROM baseline_statistics
            WHERE baseline_ref = $1 AND cmdb_id = $2 AND kpi_name = $3
            ORDER BY granularity_minutes LIMIT 1
            """,
            baseline_ref, cmdb_id, kpi_name,
        )
        if row is None:
            return None
        return BaselineStatistics(
            baseline_ref=row["baseline_ref"],
            cmdb_id=row["cmdb_id"],
            kpi_name=row["kpi_name"],
            mean=float(row["mean"]),
            stddev=float(row["stddev"]),
            p50=float(row["p50"]),
            p95=float(row["p95"]),
            p99=float(row["p99"]),
            sample_count=row["sample_count"],
            granularity_minutes=row["granularity_minutes"],
            computed_at=row["computed_at"],
        )

    async def get_time_buckets(
        self,
        baseline_ref: str,
        cmdb_id: str,
        kpi_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[BaselineTimeBucket]:
        where = "WHERE baseline_ref = $1 AND cmdb_id = $2 AND kpi_name = $3"
        args = [baseline_ref, cmdb_id, kpi_name]
        if start:
            args.append(start)
            where += f" AND bucket_start >= ${len(args)}"
        if end:
            args.append(end)
            where += f" AND bucket_start < ${len(args)}"

        rows = await self._db.fetch(
            f"""
            SELECT baseline_ref, cmdb_id, kpi_name, bucket_start,
                   mean, stddev, p50, p95, p99, sample_count
            FROM baseline_time_buckets {where}
            ORDER BY bucket_start
            """,
            *args,
        )
        return [
            BaselineTimeBucket(
                baseline_ref=r["baseline_ref"],
                cmdb_id=r["cmdb_id"],
                kpi_name=r["kpi_name"],
                bucket_start=r["bucket_start"],
                mean=float(r["mean"]),
                stddev=float(r["stddev"]),
                p50=float(r["p50"]),
                p95=float(r["p95"]),
                p99=float(r["p99"]),
                sample_count=r["sample_count"],
            )
            for r in rows
        ]

    async def delete_by_ref(self, baseline_ref: str) -> int:
        result = await self._db.execute(
            "DELETE FROM baseline_statistics WHERE baseline_ref = $1", baseline_ref
        )
        await self._db.execute(
            "DELETE FROM baseline_time_buckets WHERE baseline_ref = $1", baseline_ref
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def exists(self, baseline_ref: str) -> bool:
        val = await self._db.fetchval(
            "SELECT 1 FROM baseline_statistics WHERE baseline_ref = $1 LIMIT 1",
            baseline_ref,
        )
        return val is not None

    # Satisfy AsyncRepository contract
    async def get_by_id(self, entity_id) -> None:
        raise NotImplementedError("Use get_statistics for point lookups")

    async def save(self, entity) -> None:
        raise NotImplementedError("Use write_statistics")

    async def delete(self, entity_id) -> None:
        raise NotImplementedError("Use delete_by_ref")
