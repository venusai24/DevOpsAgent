"""Stage-level outcome enumeration."""

from enum import Enum


class StageOutcome(str, Enum):
    """High-level result of a single stage execution."""

    SUCCESS = "success"
    PARTIAL = "partial"         # Completed with caveats (e.g., low coverage).
    FORCED_EXIT = "forced_exit" # Loop guardrail terminated stage early.
    HITL_ESCALATION = "hitl_escalation"
    TIMEOUT = "timeout"
    FAILURE = "failure"
    SKIPPED = "skipped"         # Stage skipped (e.g., SUBSET reuse path).
