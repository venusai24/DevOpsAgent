"""GuardrailState — complete Phase 4 guardrail tracking sub-dict.

Spec: RecoveryAndGuardrails.md §2 (GuardrailState schema).

Nested inside InvestigationState under the key ``"guardrails"``.
Owned exclusively by the recovery layer; no LLM agent writes to it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..enums.circuit_breaker_status import CircuitBreakerStatus
from .base import BaseState, _new_uuid, _utcnow
from .retry_state import RetryRecord
from .tool_execution_state import ToolExecutionState


@dataclass(frozen=True)
class LoopGuardrailRecord(BaseState):
    """One entry per hypothesis loop exit (Domain 2)."""

    record_id: uuid.UUID = field(default_factory=_new_uuid)
    hypothesis_name: str = ""
    tool_calls_consumed: int = 0
    tool_calls_budget: int = 12
    score_at_entry: float = 0.0
    score_at_exit: float = 0.0
    # "converged" | "budget_exhausted" | "stagnant" | "forced_exit"
    exit_reason: str = "converged"
    recorded_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class CircuitBreakerSnapshot(BaseState):
    """Point-in-time CB state for reporting (Domain 4)."""

    snapshot_id: uuid.UUID = field(default_factory=_new_uuid)
    tool_name: str = ""
    state: CircuitBreakerStatus = CircuitBreakerStatus.CLOSED
    failure_count: int = 0
    last_failure_at: datetime | None = None
    open_since: datetime | None = None
    snapshotted_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class StageTimingRecord(BaseState):
    """Wall-clock timing for a single node execution."""

    node_name: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    elapsed_s: float | None = None
    # "completed" | "timeout" | "failed"
    status: str = "completed"


@dataclass(frozen=True)
class GuardrailState(BaseState):
    """All guardrail tracking data for one investigation.

    Nested in ``InvestigationState["guardrails"]``.
    Written exclusively by the orchestrator and node wrappers.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # ── Domain 1: LLM Failure Recovery ───────────────────────────────────────
    retry_log: tuple[RetryRecord, ...] = ()
    total_llm_retry_attempts: int = 0
    output_repair_attempts: int = 0

    # ── Domain 2: Loop Prevention ─────────────────────────────────────────────
    tool_call_log: tuple[ToolExecutionState, ...] = ()
    # {stage_key -> count}
    tool_calls_per_stage: dict[str, int] = field(default_factory=dict)
    loop_guardrail_log: tuple[LoopGuardrailRecord, ...] = ()
    forced_exit_triggered: bool = False
    forced_exit_reason: str | None = None

    # ── Domain 3: Trace Degradation ───────────────────────────────────────────
    # Composite 0.0–1.0 score from TraceQualityAssessor.
    trace_quality_score: float | None = None
    # Fraction of investigation_cluster covered by trace evidence.
    trace_coverage_pct: float | None = None
    # True when Stage 5.1b (Graceful Trace Degradation) is active.
    degradation_path_active: bool = False
    degradation_triggered_at: datetime | None = None
    # Human-readable reasons written to investigation_gaps.
    degradation_trigger_reasons: tuple[str, ...] = ()

    # ── Domain 4: Timeouts & Circuit Breakers ────────────────────────────────
    investigation_wall_clock_start: datetime | None = None
    stage_timings: tuple[StageTimingRecord, ...] = ()
    budget_remaining_seconds: float | None = None
    circuit_breaker_snapshots: tuple[CircuitBreakerSnapshot, ...] = ()
    global_timeout_triggered: bool = False
    stage_timeout_triggered_at: datetime | None = None
    stage_timeout_node: str | None = None

    # ── Domain 1 helpers ──────────────────────────────────────────────────────
    def append_retry(self, record: RetryRecord) -> GuardrailState:
        return self.evolve(
            retry_log=self.retry_log + (record,),
            total_llm_retry_attempts=self.total_llm_retry_attempts + 1,
        )

    def append_tool_call(self, record: ToolExecutionState) -> GuardrailState:
        updated = dict(self.tool_calls_per_stage)
        updated[record.stage_key] = updated.get(record.stage_key, 0) + 1
        return self.evolve(
            tool_call_log=self.tool_call_log + (record,),
            tool_calls_per_stage=updated,
        )

    def append_loop_record(self, record: LoopGuardrailRecord) -> GuardrailState:
        return self.evolve(loop_guardrail_log=self.loop_guardrail_log + (record,))

    def update_stage_timing(self, record: StageTimingRecord) -> GuardrailState:
        existing = {t.node_name: t for t in self.stage_timings}
        existing[record.node_name] = record
        return self.evolve(stage_timings=tuple(existing.values()))

    def update_circuit_breaker(self, snapshot: CircuitBreakerSnapshot) -> GuardrailState:
        existing = {s.tool_name: s for s in self.circuit_breaker_snapshots}
        existing[snapshot.tool_name] = snapshot
        return self.evolve(circuit_breaker_snapshots=tuple(existing.values()))
