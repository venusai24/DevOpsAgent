"""TimeoutState — wall-clock budget tracking for one investigation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .base import BaseState, _new_uuid, _utcnow


def _utcnow() -> datetime:  # noqa: F811
    return datetime.now(UTC)


@dataclass(frozen=True)
class TimeoutState(BaseState):
    """Tracks the global wall-clock budget and per-stage timeouts.

    Wraps the InvestigationClock (RecoveryAndGuardrails.md §7.1) as an
    immutable value object so that its state can be persisted in the
    checkpoint chain.

    ``hitl_pause_intervals`` accumulates the (pause_start, pause_end) pairs
    for all HITL pauses; the active investigation budget excludes these.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # Wall-clock when the investigation entered ACTIVE status.
    started_at: datetime | None = None

    # Global ceiling in seconds (from TimeoutConfig.global_timeout_seconds).
    global_budget_seconds: int = 1800

    # Accumulated HITL pause durations (excluded from budget consumption).
    hitl_pause_intervals: tuple[tuple[datetime, datetime | None], ...] = ()

    # True when global_timeout_triggered.
    global_timeout_triggered: bool = False
    global_timeout_triggered_at: datetime | None = None

    # Per-node timeout events (node_name -> triggered_at).
    node_timeout_events: dict[str, str] = field(default_factory=dict)  # ISO 8601 values

    @property
    def total_hitl_pause_seconds(self) -> float:
        total = 0.0
        for start, end in self.hitl_pause_intervals:
            if end is not None:
                total += (end - start).total_seconds()
        return total

    @property
    def elapsed_active_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        raw = (_utcnow() - self.started_at).total_seconds()
        return max(0.0, raw - self.total_hitl_pause_seconds)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.global_budget_seconds - self.elapsed_active_seconds)

    @property
    def is_expired(self) -> bool:
        return self.remaining_seconds <= 0.0

    def start(self, at: datetime | None = None) -> TimeoutState:
        return self.evolve(started_at=at or _utcnow())

    def pause_hitl(self, at: datetime | None = None) -> TimeoutState:
        ts = at or _utcnow()
        return self.evolve(
            hitl_pause_intervals=self.hitl_pause_intervals + ((ts, None),)
        )

    def resume_hitl(self, at: datetime | None = None) -> TimeoutState:
        ts = at or _utcnow()
        intervals = list(self.hitl_pause_intervals)
        if intervals and intervals[-1][1] is None:
            intervals[-1] = (intervals[-1][0], ts)
        return self.evolve(hitl_pause_intervals=tuple(intervals))

    def trigger_global(self, at: datetime | None = None) -> TimeoutState:
        ts = at or _utcnow()
        return self.evolve(
            global_timeout_triggered=True,
            global_timeout_triggered_at=ts,
        )

    def record_node_timeout(self, node_name: str,
                             at: datetime | None = None) -> TimeoutState:
        ts = (at or _utcnow()).isoformat()
        updated = dict(self.node_timeout_events)
        updated[node_name] = ts
        return self.evolve(node_timeout_events=updated)
