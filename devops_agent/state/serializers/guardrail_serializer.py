"""Serializer for GuardrailState and its nested types."""

from __future__ import annotations

from typing import Any

from ..enums.circuit_breaker_status import CircuitBreakerStatus
from ..enums.failure_type import FailureType
from ..models.base import StateMetadata, StateVersion
from ..models.guardrail_state import (
    CircuitBreakerSnapshot,
    GuardrailState,
    LoopGuardrailRecord,
    StageTimingRecord,
)
from ..models.retry_state import RetryRecord
from ..models.tool_execution_state import ToolExecutionState
from .base_serializer import BaseSerializer
from .codec import decode_datetime, decode_enum, decode_uuid, encode


class RetryRecordSerializer(BaseSerializer[RetryRecord]):
    def serialise(self, s: RetryRecord) -> dict[str, Any]:
        return {
            "retry_record_id": str(s.retry_record_id),
            "node_name": s.node_name,
            "failure_type": s.failure_type.value,
            "attempt_number": s.attempt_number,
            "error_code": s.error_code,
            "repair_prompt_injected": s.repair_prompt_injected,
            "hint_injected": s.hint_injected,
            "backoff_seconds": s.backoff_seconds,
            "attempted_at": encode(s.attempted_at),
            "resolution": s.resolution,
        }

    def deserialise(self, d: dict[str, Any]) -> RetryRecord:
        return RetryRecord(
            retry_record_id=decode_uuid(d["retry_record_id"]),
            node_name=d.get("node_name", ""),
            failure_type=decode_enum(FailureType, d.get("failure_type", "schema_violation")),
            attempt_number=d.get("attempt_number", 0),
            error_code=d.get("error_code"),
            repair_prompt_injected=d.get("repair_prompt_injected", False),
            hint_injected=d.get("hint_injected", False),
            backoff_seconds=d.get("backoff_seconds", 0.0),
            attempted_at=decode_datetime(d.get("attempted_at")),
            resolution=d.get("resolution", "success"),
        )


class ToolExecutionStateSerializer(BaseSerializer[ToolExecutionState]):
    def serialise(self, s: ToolExecutionState) -> dict[str, Any]:
        return {
            "tool_execution_id": str(s.tool_execution_id),
            "investigation_id": str(s.investigation_id),
            "node_name": s.node_name,
            "stage_key": s.stage_key,
            "hypothesis_name": s.hypothesis_name,
            "tool_name": s.tool_name,
            "parameters": s.parameters,
            "invoked_at": encode(s.invoked_at),
            "completed_at": encode(s.completed_at),
            "elapsed_ms": s.elapsed_ms,
            "retry_attempt": s.retry_attempt,
            "succeeded": s.succeeded,
            "error_code": s.error_code,
            "failure_type": s.failure_type.value if s.failure_type else None,
            "output_summary": s.output_summary,
            "circuit_breaker_rejected": s.circuit_breaker_rejected,
        }

    def deserialise(self, d: dict[str, Any]) -> ToolExecutionState:
        return ToolExecutionState(
            tool_execution_id=decode_uuid(d["tool_execution_id"]),
            investigation_id=decode_uuid(d["investigation_id"]),
            node_name=d.get("node_name", ""),
            stage_key=d.get("stage_key", ""),
            hypothesis_name=d.get("hypothesis_name"),
            tool_name=d.get("tool_name", ""),
            parameters=d.get("parameters", {}),
            invoked_at=decode_datetime(d.get("invoked_at")),
            completed_at=decode_datetime(d.get("completed_at")),
            elapsed_ms=d.get("elapsed_ms"),
            retry_attempt=d.get("retry_attempt", 0),
            succeeded=d.get("succeeded", False),
            error_code=d.get("error_code"),
            failure_type=decode_enum(FailureType, d.get("failure_type")) if d.get("failure_type") else None,
            output_summary=d.get("output_summary"),
            circuit_breaker_rejected=d.get("circuit_breaker_rejected", False),
        )


_rr_ser = RetryRecordSerializer()
_te_ser = ToolExecutionStateSerializer()


class GuardrailStateSerializer(BaseSerializer[GuardrailState]):
    """Full round-trip serializer for GuardrailState."""

    def serialise(self, s: GuardrailState) -> dict[str, Any]:
        return {
            "investigation_id": str(s.investigation_id),
            "version": s.version.value,
            # D1
            "retry_log": [_rr_ser.serialise(r) for r in s.retry_log],
            "total_llm_retry_attempts": s.total_llm_retry_attempts,
            "output_repair_attempts": s.output_repair_attempts,
            # D2
            "tool_call_log": [_te_ser.serialise(t) for t in s.tool_call_log],
            "tool_calls_per_stage": dict(s.tool_calls_per_stage),
            "loop_guardrail_log": [
                {
                    "record_id": str(r.record_id),
                    "hypothesis_name": r.hypothesis_name,
                    "tool_calls_consumed": r.tool_calls_consumed,
                    "tool_calls_budget": r.tool_calls_budget,
                    "score_at_entry": r.score_at_entry,
                    "score_at_exit": r.score_at_exit,
                    "exit_reason": r.exit_reason,
                    "recorded_at": encode(r.recorded_at),
                }
                for r in s.loop_guardrail_log
            ],
            "forced_exit_triggered": s.forced_exit_triggered,
            "forced_exit_reason": s.forced_exit_reason,
            # D3
            "trace_quality_score": s.trace_quality_score,
            "trace_coverage_pct": s.trace_coverage_pct,
            "degradation_path_active": s.degradation_path_active,
            "degradation_triggered_at": encode(s.degradation_triggered_at),
            "degradation_trigger_reasons": list(s.degradation_trigger_reasons),
            # D4
            "investigation_wall_clock_start": encode(s.investigation_wall_clock_start),
            "stage_timings": [
                {
                    "node_name": t.node_name,
                    "started_at": encode(t.started_at),
                    "ended_at": encode(t.ended_at),
                    "elapsed_s": t.elapsed_s,
                    "status": t.status,
                }
                for t in s.stage_timings
            ],
            "budget_remaining_seconds": s.budget_remaining_seconds,
            "circuit_breaker_snapshots": [
                {
                    "snapshot_id": str(c.snapshot_id),
                    "tool_name": c.tool_name,
                    "state": c.state.value,
                    "failure_count": c.failure_count,
                    "last_failure_at": encode(c.last_failure_at),
                    "open_since": encode(c.open_since),
                    "snapshotted_at": encode(c.snapshotted_at),
                }
                for c in s.circuit_breaker_snapshots
            ],
            "global_timeout_triggered": s.global_timeout_triggered,
            "stage_timeout_triggered_at": encode(s.stage_timeout_triggered_at),
            "stage_timeout_node": s.stage_timeout_node,
            # Base
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> GuardrailState:
        loop_log = tuple(
            LoopGuardrailRecord(
                record_id=decode_uuid(r["record_id"]),
                hypothesis_name=r.get("hypothesis_name", ""),
                tool_calls_consumed=r.get("tool_calls_consumed", 0),
                tool_calls_budget=r.get("tool_calls_budget", 12),
                score_at_entry=r.get("score_at_entry", 0.0),
                score_at_exit=r.get("score_at_exit", 0.0),
                exit_reason=r.get("exit_reason", "converged"),
                recorded_at=decode_datetime(r.get("recorded_at")),
            )
            for r in d.get("loop_guardrail_log", [])
        )
        stage_timings = tuple(
            StageTimingRecord(
                node_name=t.get("node_name", ""),
                started_at=decode_datetime(t.get("started_at")),
                ended_at=decode_datetime(t.get("ended_at")),
                elapsed_s=t.get("elapsed_s"),
                status=t.get("status", "completed"),
            )
            for t in d.get("stage_timings", [])
        )
        cb_snapshots = tuple(
            CircuitBreakerSnapshot(
                snapshot_id=decode_uuid(c["snapshot_id"]),
                tool_name=c.get("tool_name", ""),
                state=decode_enum(CircuitBreakerStatus, c.get("state", "closed")),
                failure_count=c.get("failure_count", 0),
                last_failure_at=decode_datetime(c.get("last_failure_at")),
                open_since=decode_datetime(c.get("open_since")),
                snapshotted_at=decode_datetime(c.get("snapshotted_at")),
            )
            for c in d.get("circuit_breaker_snapshots", [])
        )
        meta_raw = d.get("metadata") or {}
        metadata = StateMetadata(
            created_at=decode_datetime(meta_raw.get("created_at")),
            updated_at=decode_datetime(meta_raw.get("updated_at")),
            producing_node=meta_raw.get("producing_node"),
            checkpoint_id=meta_raw.get("checkpoint_id"),
            schema_version=meta_raw.get("schema_version", 1),
        )
        return GuardrailState(
            version=StateVersion(value=d.get("version", 0)),
            metadata=metadata,
            investigation_id=decode_uuid(d["investigation_id"]),
            retry_log=tuple(_rr_ser.deserialise(r) for r in d.get("retry_log", [])),
            total_llm_retry_attempts=d.get("total_llm_retry_attempts", 0),
            output_repair_attempts=d.get("output_repair_attempts", 0),
            tool_call_log=tuple(_te_ser.deserialise(t) for t in d.get("tool_call_log", [])),
            tool_calls_per_stage=d.get("tool_calls_per_stage", {}),
            loop_guardrail_log=loop_log,
            forced_exit_triggered=d.get("forced_exit_triggered", False),
            forced_exit_reason=d.get("forced_exit_reason"),
            trace_quality_score=d.get("trace_quality_score"),
            trace_coverage_pct=d.get("trace_coverage_pct"),
            degradation_path_active=d.get("degradation_path_active", False),
            degradation_triggered_at=decode_datetime(d.get("degradation_triggered_at")),
            degradation_trigger_reasons=tuple(d.get("degradation_trigger_reasons", [])),
            investigation_wall_clock_start=decode_datetime(d.get("investigation_wall_clock_start")),
            stage_timings=stage_timings,
            budget_remaining_seconds=d.get("budget_remaining_seconds"),
            circuit_breaker_snapshots=cb_snapshots,
            global_timeout_triggered=d.get("global_timeout_triggered", False),
            stage_timeout_triggered_at=decode_datetime(d.get("stage_timeout_triggered_at")),
            stage_timeout_node=d.get("stage_timeout_node"),
        )
