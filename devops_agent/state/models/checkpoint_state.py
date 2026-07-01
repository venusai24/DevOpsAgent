"""CheckpointState — metadata for one LangGraph checkpoint snapshot.

Mirrors the structure of PostgresSaver's checkpoint rows while adding our
own audit fields.  The actual ``channel_values`` payload (serialised
InvestigationState) is stored as a JSON column in the checkpoint table and
is not represented as a Python field here to keep the model thin.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class CheckpointState(BaseState):
    """Metadata for one point-in-time snapshot of InvestigationState.

    The checkpoint chain is maintained by ``parent_checkpoint_id``; this
    enables full replay from any historical snapshot.

    The ``payload`` field carries the serialised InvestigationState dict.
    It is stored as JSONB in Postgres; the serialization layer is responsible
    for the round-trip.
    """

    checkpoint_id: uuid.UUID = field(default_factory=_new_uuid)

    # Always equals the investigation_id / LangGraph thread_id.
    thread_id: uuid.UUID = field(default_factory=_new_uuid)

    # Predecessor in the chain; None only for the first checkpoint.
    parent_checkpoint_id: uuid.UUID | None = None

    # The LangGraph node whose completion produced this checkpoint.
    producing_node: str = ""

    # Monotonically increasing within a thread.
    sequence_number: int = 0

    # When this checkpoint was written.
    created_at: datetime = field(default_factory=_utcnow)

    # The full InvestigationState serialised to a JSON-safe dict.
    # Persisted as JSONB; loaded on demand by CheckpointManager.
    payload: dict[str, Any] | None = None

    # Schema version of the payload for migration compatibility.
    payload_schema_version: int = 1

    # True if this is the most recent checkpoint for the thread.
    is_latest: bool = True

    # "active" | "hitl_paused" | "completed" | "rolled_back"
    checkpoint_status: str = "active"

    # LangGraph-internal channel_versions vector (opaque dict).
    channel_versions: dict[str, Any] = field(default_factory=dict)

    # LangGraph-internal pending_writes log (opaque list).
    pending_writes: list[Any] = field(default_factory=list)

    def as_historical(self) -> CheckpointState:
        """Return a copy marked as no longer the latest checkpoint."""
        return self.evolve(is_latest=False)

    def with_payload(self, payload: dict[str, Any]) -> CheckpointState:
        return self.evolve(payload=payload)
