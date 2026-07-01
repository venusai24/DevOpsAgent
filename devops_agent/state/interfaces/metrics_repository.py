"""Abstract repository for ExecutionMetrics persistence."""

from __future__ import annotations

import uuid
from abc import abstractmethod
from collections.abc import Sequence

from ..models.execution_metrics import ExecutionMetrics
from .repository import AsyncRepository


class MetricsRepository(AsyncRepository[ExecutionMetrics]):
    """Persists execution metrics for analytics and dashboards (Phase 3 §3.7)."""

    @abstractmethod
    async def get_by_investigation(
        self, investigation_id: uuid.UUID
    ) -> ExecutionMetrics | None:
        """Return metrics for a specific investigation."""

    @abstractmethod
    async def upsert(self, metrics: ExecutionMetrics) -> ExecutionMetrics:
        """Upsert metrics for an investigation."""

    @abstractmethod
    async def find_by_confidence(
        self, confidence_level: str, limit: int = 100
    ) -> Sequence[ExecutionMetrics]:
        """Return investigations that reached a given confidence level."""
