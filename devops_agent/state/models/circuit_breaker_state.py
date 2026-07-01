"""CircuitBreakerState — per-tool circuit breaker FSM state.

Spec: RecoveryAndGuardrails.md §7.3.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..enums.circuit_breaker_status import CircuitBreakerStatus
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class CircuitBreakerState(BaseState):
    """Immutable FSM state for one tool's circuit breaker.

    One row per (tool_name) in the circuit_breaker_state table.
    The CircuitBreakerRegistry reads and writes these via the
    PostgresCircuitBreakerRepository.

    Transitions are enforced by CircuitBreaker.call() in the business layer;
    this model is pure data.
    """

    cb_id: uuid.UUID = field(default_factory=_new_uuid)

    # Tool name as registered in the tool registry.
    tool_name: str = ""

    status: CircuitBreakerStatus = CircuitBreakerStatus.CLOSED

    # Number of consecutive failures while in CLOSED state.
    failure_count: int = 0

    # Threshold from CircuitBreakerConfig.failure_threshold.
    failure_threshold: int = 5

    # When the breaker most recently transitioned to OPEN.
    opened_at: datetime | None = None

    # When the breaker most recently transitioned to HALF_OPEN.
    half_open_since: datetime | None = None

    # Probe interval from CircuitBreakerConfig.half_open_probe_interval_s.
    open_timeout_s: int = 120

    # Total lifetime transition counts (for telemetry).
    total_open_transitions: int = 0
    total_close_transitions: int = 0

    last_failure_at: datetime | None = None
    last_success_at: datetime | None = None

    # ── FSM transition helpers ────────────────────────────────────────────────

    def record_failure(self) -> CircuitBreakerState:
        now = _utcnow()
        new_count = self.failure_count + 1
        if new_count >= self.failure_threshold and self.status == CircuitBreakerStatus.CLOSED:
            return self.evolve(
                failure_count=new_count,
                last_failure_at=now,
                status=CircuitBreakerStatus.OPEN,
                opened_at=now,
                total_open_transitions=self.total_open_transitions + 1,
            )
        if self.status == CircuitBreakerStatus.HALF_OPEN:
            # Probe failed — back to OPEN.
            return self.evolve(
                failure_count=new_count,
                last_failure_at=now,
                status=CircuitBreakerStatus.OPEN,
                opened_at=now,
                half_open_since=None,
                total_open_transitions=self.total_open_transitions + 1,
            )
        return self.evolve(failure_count=new_count, last_failure_at=now)

    def record_success(self) -> CircuitBreakerState:
        now = _utcnow()
        if self.status in (CircuitBreakerStatus.HALF_OPEN, CircuitBreakerStatus.OPEN):
            return self.evolve(
                failure_count=0,
                last_success_at=now,
                status=CircuitBreakerStatus.CLOSED,
                opened_at=None,
                half_open_since=None,
                total_close_transitions=self.total_close_transitions + 1,
            )
        return self.evolve(failure_count=0, last_success_at=now)

    def try_half_open(self, now: datetime | None = None) -> CircuitBreakerState:
        """Transition to HALF_OPEN if the open timeout has elapsed."""
        if self.status != CircuitBreakerStatus.OPEN or self.opened_at is None:
            return self
        ts = now or _utcnow()
        if (ts - self.opened_at).total_seconds() >= self.open_timeout_s:
            return self.evolve(
                status=CircuitBreakerStatus.HALF_OPEN,
                half_open_since=ts,
            )
        return self

    @property
    def should_reject_call(self) -> bool:
        return self.status == CircuitBreakerStatus.OPEN
