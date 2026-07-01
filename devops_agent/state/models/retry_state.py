"""RetryState — tracks retry attempt counts and backoff for one node."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..enums.failure_type import FailureType
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class RetryRecord(BaseState):
    """One retry attempt entry.  Appended to RetryState.retry_log."""

    retry_record_id: uuid.UUID = field(default_factory=_new_uuid)
    node_name: str = ""
    failure_type: FailureType = FailureType.SCHEMA_VIOLATION
    attempt_number: int = 0
    error_code: str | None = None
    repair_prompt_injected: bool = False
    hint_injected: bool = False
    backoff_seconds: float = 0.0
    attempted_at: datetime = field(default_factory=_utcnow)
    # "success" | "hitl_escalation" | "graceful_degrade" | "fatal"
    resolution: str = "success"


@dataclass(frozen=True)
class RetryState(BaseState):
    """Accumulates retry records for one investigation.

    Owned by the RetryCoordinator (Domain 1).  Persisted inside GuardrailState
    and therefore inside the LangGraph checkpoint chain.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # All retry attempts across all nodes.
    retry_log: tuple[RetryRecord, ...] = ()

    # Aggregate counters (derived from retry_log but cached here to avoid
    # iterating the log on every read).
    total_tool_retries: int = 0
    total_schema_retries: int = 0
    total_output_repair_attempts: int = 0

    def append(self, record: RetryRecord) -> RetryState:
        """Return a copy with the record appended and counters updated."""
        tool_delta = 1 if record.failure_type.is_tool_level else 0
        schema_delta = 1 if record.failure_type in (
            FailureType.SCHEMA_VIOLATION,
            FailureType.FIELD_TYPE_ERROR,
            FailureType.MISSING_REQUIRED_FIELD,
        ) else 0
        repair_delta = 1 if record.failure_type == FailureType.UNPARSEABLE_JSON else 0
        return self.evolve(
            retry_log=self.retry_log + (record,),
            total_tool_retries=self.total_tool_retries + tool_delta,
            total_schema_retries=self.total_schema_retries + schema_delta,
            total_output_repair_attempts=self.total_output_repair_attempts + repair_delta,
        )
