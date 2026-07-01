"""ContextSnapshot — lightweight summary of InvestigationState at a given checkpoint.

Used by:
  - The Stage 6 topology-manifest staleness check (Phase 3 §3.5.2).
  - HITL resume hydration to verify the graph can safely continue.
  - The ResumeManager to detect schema-version mismatches before deserialising
    the full checkpoint payload.

This is intentionally a thin, fast-to-deserialise summary — not the full
InvestigationState blob.  The full payload lives in CheckpointState.payload.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..enums.investigation_status import InvestigationStatus
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class ContextSnapshot(BaseState):
    """Thin summary of InvestigationState captured at checkpoint creation."""

    snapshot_id: uuid.UUID = field(default_factory=_new_uuid)

    # Thread identity.
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)
    checkpoint_id: uuid.UUID | None = None

    # Status at the time of the snapshot.
    status: InvestigationStatus = InvestigationStatus.CLAIMED

    # The node that produced this checkpoint.
    producing_node: str = ""

    # Stage progress indicator.
    last_completed_stage: str = ""

    # Topology manifest version at snapshot time (for §3.5.2 staleness check).
    topology_manifest_version: str | None = None

    # Hypothesis and evidence progress (sparse — populated from Stage 5 onward).
    hypothesis_count: int = 0
    surviving_hypothesis_count: int = 0
    best_hypothesis_score: float | None = None

    # Guardrail summary (so the ResumeManager can decide whether restart is safer).
    forced_exit_triggered: bool = False
    trace_degradation_active: bool = False
    global_timeout_triggered: bool = False
    total_retries: int = 0

    # Schema version of the full CheckpointState.payload.
    payload_schema_version: int = 1

    snapshotted_at: datetime = field(default_factory=_utcnow)

    @property
    def is_resumable(self) -> bool:
        """Return True when the investigation can safely be resumed."""
        return (
            self.status in (
                InvestigationStatus.HITL_PAUSED,
                InvestigationStatus.ACTIVE,
                InvestigationStatus.PENDING_CONTEXT,
            )
            and not self.global_timeout_triggered
        )
