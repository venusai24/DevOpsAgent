"""ExecutionState — top-level runtime state for a single graph invocation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..enums.investigation_status import InvestigationStatus
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class ExecutionState(BaseState):
    """Tracks the top-level lifecycle of one graph.invoke() call.

    One ExecutionState row corresponds to one investigation.  The
    ``investigation_id`` equals the LangGraph ``thread_id`` as mandated by
    ADR-002 §9.1.

    This is the root state object persisted in the Investigation Registry
    (control-plane).  The full ``InvestigationState`` payload lives in the
    LangGraph Checkpointer (data-plane).
    """

    # Primary key; equals LangGraph thread_id.
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # Topology domain known before Stage -1 runs (see Phase 3 §3.1.4).
    cluster_id: str = ""

    # Hash of sorted tc_values + sorted cmdb_ids — exact-duplicate fast path.
    symptom_signature: str = ""

    # ISO 8601 range [start, end) captured from the alert payload.
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None

    status: InvestigationStatus = InvestigationStatus.CLAIMED

    # Wall-clock when graph.invoke() was called.
    started_at: datetime = field(default_factory=_utcnow)

    # Populated when status transitions to a terminal value.
    completed_at: datetime | None = None

    # The LangGraph node currently executing (denormalised; for dashboards).
    current_node: str | None = None

    # ── Dedup relationship fields (Phase 3 §3.1.3) ───────────────────────────
    dedup_decision: str | None = None
    related_investigation_id: uuid.UUID | None = None
    topology_manifest_version: str | None = None
    stage0_cache_key: str | None = None

    # ── Terminal audit fields ─────────────────────────────────────────────────
    confidence_level: str | None = None
    root_cause_cmdb_id: str | None = None

    def with_status(self, status: InvestigationStatus,
                    node: str | None = None) -> ExecutionState:
        """Return a copy with status updated and completed_at set if terminal."""
        completed = (
            _utcnow() if status.is_terminal and self.completed_at is None
            else self.completed_at
        )
        return self.evolve(
            status=status,
            current_node=node or self.current_node,
            completed_at=completed,
        )
