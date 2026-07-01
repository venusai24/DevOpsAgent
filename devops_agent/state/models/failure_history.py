"""FailureHistory — durable log of all failure events for one investigation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..enums.failure_type import FailureType
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class FailureEvent(BaseState):
    """One failure event in the investigation history."""

    event_id: uuid.UUID = field(default_factory=_new_uuid)
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    failure_type: FailureType = FailureType.SCHEMA_VIOLATION
    node_name: str = ""
    stage_key: str = ""
    tool_name: str | None = None
    error_code: str | None = None

    # Error message or model output snippet (truncated to 2000 chars).
    error_detail: str | None = None

    occurred_at: datetime = field(default_factory=_utcnow)

    # How many retry attempts had been made before this failure.
    retry_attempt_at_failure: int = 0

    # "retried" | "repaired" | "escalated" | "degraded" | "fatal"
    resolution: str = "retried"


@dataclass(frozen=True)
class FailureHistory(BaseState):
    """Complete failure event log for one investigation.

    Distinct from RetryState (which tracks retry mechanics) and
    RecoveryState (which tracks recovery actions).  FailureHistory is the
    raw event stream; the other two models are derived views.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)
    events: tuple[FailureEvent, ...] = ()

    # Counts by failure type (derived but cached).
    counts_by_type: dict[str, int] = field(default_factory=dict)

    def append(self, event: FailureEvent) -> FailureHistory:
        updated = dict(self.counts_by_type)
        key = event.failure_type.value
        updated[key] = updated.get(key, 0) + 1
        return self.evolve(
            events=self.events + (event,),
            counts_by_type=updated,
        )

    def events_for_node(self, node_name: str) -> tuple[FailureEvent, ...]:
        return tuple(e for e in self.events if e.node_name == node_name)

    def events_for_type(self, ftype: FailureType) -> tuple[FailureEvent, ...]:
        return tuple(e for e in self.events if e.failure_type == ftype)
