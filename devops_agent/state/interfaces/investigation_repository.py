"""Abstract repository for the Investigation Registry (control plane).

Spec: StateManagement&PersistenceLayer.md §3.1 and §3.2.
"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from collections.abc import Sequence
from datetime import datetime

from ..enums.dedup_decision import DedupDecision
from ..enums.investigation_status import InvestigationStatus
from ..models.execution_state import ExecutionState
from .repository import AsyncRepository


class InvestigationRepository(AsyncRepository[ExecutionState]):
    """Controls read/write access to investigation_registry.

    All writes that need dedup-safe concurrency must go through
    ``claim_with_dedup``, which opens a SERIALIZABLE transaction.
    Point-status updates (e.g., marking hitl_paused) use a lightweight
    ``update_status`` that runs at READ COMMITTED.
    """

    @abstractmethod
    async def claim_with_dedup(
        self,
        investigation_id: uuid.UUID,
        cluster_id: str,
        symptom_signature: str,
        time_range_start: datetime,
        time_range_end: datetime,
    ) -> tuple[ExecutionState, DedupDecision, uuid.UUID | None]:
        """Atomically insert a registry row and classify the dedup decision.

        Uses SERIALIZABLE isolation (Phase 3 §3.2.2) to catch the phantom-read
        race between two concurrent NEW claims.

        Returns:
            (execution_state, decision, related_investigation_id)
        Raises:
            DedupConflictExhausted: after MAX_DEDUP_RETRIES SERIALIZABLE failures.
        """

    @abstractmethod
    async def update_status(
        self,
        investigation_id: uuid.UUID,
        status: InvestigationStatus,
        current_node: str | None = None,
        expected_version: int | None = None,
    ) -> ExecutionState:
        """Transition the status of a registry row.

        If ``expected_version`` is provided, uses optimistic-concurrency CAS
        and raises ``ConcurrentModificationError`` on mismatch.
        """

    @abstractmethod
    async def find_overlapping(
        self,
        cluster_id: str,
        time_range_start: datetime,
        time_range_end: datetime,
        exclude_terminal: bool = True,
    ) -> Sequence[ExecutionState]:
        """Return all registry rows whose time_range overlaps the given window."""

    @abstractmethod
    async def find_by_status(
        self,
        status: InvestigationStatus,
        cluster_id: str | None = None,
        limit: int = 100,
    ) -> Sequence[ExecutionState]:
        """Return investigations matching the given status, ordered by created_at."""

    @abstractmethod
    async def mark_superseded(
        self,
        investigation_ids: Sequence[uuid.UUID],
        superseded_by: uuid.UUID,
    ) -> None:
        """Batch-update status to SUPERSEDED for investigations subsumed by a SUPERSET."""

    @abstractmethod
    async def update_terminal_fields(
        self,
        investigation_id: uuid.UUID,
        confidence_level: str | None,
        root_cause_cmdb_id: str | None,
    ) -> None:
        """Populate audit fields at report delivery time (Stage 8)."""

    @abstractmethod
    async def find_stale_hitl(self, older_than: datetime) -> Sequence[ExecutionState]:
        """Return hitl_paused investigations whose updated_at predates ``older_than``.

        Used by the watchdog job described in Phase 3 §3.3.4.
        """
