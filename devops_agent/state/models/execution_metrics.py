"""ExecutionMetrics — aggregate performance and cost metrics for one investigation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseState, _new_uuid


@dataclass(frozen=True)
class StageMetrics(BaseState):
    """Per-stage metrics for one investigation."""

    stage_key: str = ""
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    retry_count: int = 0
    token_spend: int = 0
    elapsed_s: float | None = None


@dataclass(frozen=True)
class ExecutionMetrics(BaseState):
    """Aggregate metrics for one investigation lifecycle.

    Persisted in the Investigation Registry for analytics (Phase 3 §3.7).
    Never read by the RCA agents — ops/dashboard use only.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # Total wall-clock time (seconds), excluding HITL pauses.
    total_active_seconds: float | None = None
    total_hitl_pause_seconds: float | None = None

    # Aggregate token spend across all agents and stages.
    total_token_spend: int = 0

    # Aggregate tool call counts.
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    circuit_breaker_rejections: int = 0

    # Retry aggregates.
    total_retries: int = 0
    total_schema_repairs: int = 0
    total_hitl_escalations: int = 0

    # Whether the trace degradation path was active.
    trace_degradation_active: bool = False

    # Whether any forced exit occurred.
    forced_exit_occurred: bool = False

    # Per-stage breakdown.
    stage_metrics: tuple[StageMetrics, ...] = ()

    # Final confidence level (populated at Stage 7).
    final_confidence_level: str | None = None

    computed_at: datetime | None = None

    def add_stage(self, metrics: StageMetrics) -> ExecutionMetrics:
        existing = {s.stage_key: s for s in self.stage_metrics}
        existing[metrics.stage_key] = metrics
        return self.evolve(
            stage_metrics=tuple(existing.values()),
            total_tool_calls=self.total_tool_calls + metrics.tool_calls,
            successful_tool_calls=self.successful_tool_calls + metrics.successful_tool_calls,
            failed_tool_calls=self.failed_tool_calls + metrics.failed_tool_calls,
            total_retries=self.total_retries + metrics.retry_count,
            total_token_spend=self.total_token_spend + metrics.token_spend,
        )
