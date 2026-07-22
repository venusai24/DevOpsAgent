"""Circuit breaker finite-state-machine states (RecoveryAndGuardrails.md §7.3)."""

from enum import StrEnum


class CircuitBreakerStatus(StrEnum):
    """The three states of the per-tool circuit breaker FSM.

    Transitions:
        CLOSED  → OPEN:      failure_count >= threshold (5 consecutive failures)
        OPEN    → HALF_OPEN: open_timeout_s (120 s) elapsed since OPEN transition
        HALF_OPEN → CLOSED:  probe call succeeds
        HALF_OPEN → OPEN:    probe call fails
    """

    CLOSED = "closed"       # Normal operation; all calls pass through.
    OPEN = "open"           # All calls rejected immediately (fast-fail).
    HALF_OPEN = "half_open" # One probe call allowed; result determines transition.
