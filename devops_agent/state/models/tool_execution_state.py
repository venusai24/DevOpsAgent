"""ToolExecutionState — immutable record of a single tool call."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums.failure_type import FailureType
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class ToolExecutionState(BaseState):
    """Immutable record for one tool invocation, successful or failed.

    Stored as part of GuardrailState.tool_call_log.  The persistence layer
    appends these records; no record is ever mutated after creation.
    """

    tool_execution_id: uuid.UUID = field(default_factory=_new_uuid)
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # Which LangGraph node triggered this call.
    node_name: str = ""

    # ADR-001 stage the call belongs to (e.g., "stage_5_evidence_collection").
    stage_key: str = ""

    # Hypothesis name if this call belongs to Stage 5 evidence collection.
    hypothesis_name: str | None = None

    tool_name: str = ""

    # Serialised parameters passed to the tool (JSON-safe dict).
    parameters: dict[str, Any] = field(default_factory=dict)

    invoked_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None
    elapsed_ms: float | None = None

    # 0 = first attempt; increments on each retry for the same logical call.
    retry_attempt: int = 0

    succeeded: bool = False

    # Error code from Phase 2 §1.4 (e.g., "UNKNOWN_CMDB_ID"), if any.
    error_code: str | None = None
    failure_type: FailureType | None = None

    # Short summary of the tool output (never the raw payload).
    output_summary: str | None = None

    # True when the circuit breaker was OPEN and the call was fast-failed.
    circuit_breaker_rejected: bool = False

    def close(self, succeeded: bool,
              error_code: str | None = None,
              failure_type: FailureType | None = None,
              output_summary: str | None = None) -> ToolExecutionState:
        """Return a completed copy of this record."""
        now = _utcnow()
        elapsed = (now - self.invoked_at).total_seconds() * 1000
        return self.evolve(
            completed_at=now,
            elapsed_ms=elapsed,
            succeeded=succeeded,
            error_code=error_code,
            failure_type=failure_type,
            output_summary=output_summary,
        )
