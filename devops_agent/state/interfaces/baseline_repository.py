"""Abstract repository for the Baseline Statistics Store (Phase 3 §3.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime


class BaselineStatistics:
    """DTO for one (baseline_ref, cmdb_id, kpi_name) baseline record."""

    __slots__ = (
        "baseline_ref", "cmdb_id", "kpi_name", "mean", "stddev",
        "p50", "p95", "p99", "sample_count", "granularity_minutes",
        "computed_at",
    )

    def __init__(
        self,
        baseline_ref: str,
        cmdb_id: str,
        kpi_name: str,
        mean: float,
        stddev: float,
        p50: float,
        p95: float,
        p99: float,
        sample_count: int,
        granularity_minutes: int,
        computed_at: datetime | None = None,
    ):
        self.baseline_ref = baseline_ref
        self.cmdb_id = cmdb_id
        self.kpi_name = kpi_name
        self.mean = mean
        self.stddev = stddev
        self.p50 = p50
        self.p95 = p95
        self.p99 = p99
        self.sample_count = sample_count
        self.granularity_minutes = granularity_minutes
        self.computed_at = computed_at


class BaselineTimeBucket:
    """DTO for one time-bucket row in the baseline store."""

    __slots__ = (
        "baseline_ref", "cmdb_id", "kpi_name", "bucket_start",
        "mean", "stddev", "p50", "p95", "p99", "sample_count",
    )

    def __init__(
        self,
        baseline_ref: str,
        cmdb_id: str,
        kpi_name: str,
        bucket_start: datetime,
        mean: float,
        stddev: float,
        p50: float,
        p95: float,
        p99: float,
        sample_count: int,
    ):
        self.baseline_ref = baseline_ref
        self.cmdb_id = cmdb_id
        self.kpi_name = kpi_name
        self.bucket_start = bucket_start
        self.mean = mean
        self.stddev = stddev
        self.p50 = p50
        self.p95 = p95
        self.p99 = p99
        self.sample_count = sample_count


class BaselineRepository(ABC):
    """Controls read/write access to the Baseline Statistics Store.

    Writes:
        idempotent (identical input hash returns existing baseline_ref).
    Reads:
        narrow point/range lookups by (baseline_ref, cmdb_id, kpi_name).

    Retention is lifecycle-based (Phase 3 §3.4.4), NOT TTL-based.
    """

    @abstractmethod
    async def write_statistics(
        self,
        baseline_ref: str,
        rows: Sequence[BaselineStatistics],
    ) -> str:
        """Idempotently write a batch of statistics rows.

        Returns the canonical baseline_ref (may equal the supplied one for
        de-duplicated writes).
        """

    @abstractmethod
    async def get_statistics(
        self,
        baseline_ref: str,
        cmdb_id: str,
        kpi_name: str,
    ) -> BaselineStatistics | None:
        """Point lookup for aggregate statistics."""

    @abstractmethod
    async def get_time_buckets(
        self,
        baseline_ref: str,
        cmdb_id: str,
        kpi_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[BaselineTimeBucket]:
        """Range query over time-bucket rows."""

    @abstractmethod
    async def delete_by_ref(self, baseline_ref: str) -> int:
        """Delete all rows for a baseline_ref.  Returns count deleted.

        Called by the retention job (Phase 3 §3.4.4) only after all
        referencing investigations are terminal.
        """

    @abstractmethod
    async def exists(self, baseline_ref: str) -> bool:
        """Return True when the baseline_ref has at least one row."""
