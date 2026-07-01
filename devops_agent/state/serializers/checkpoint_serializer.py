"""Serializers for RecoveryState, TimeoutState, FailureHistory,
FallbackHistory, ExecutionMetrics, and CheckpointState.

All are standalone BaseSerializer subclasses importable without circular deps.
"""

from __future__ import annotations

from typing import Any

from ..enums.failure_type import FailureType
from ..models.base import StateMetadata, StateVersion
from ..models.checkpoint_state import CheckpointState
from ..models.execution_metrics import ExecutionMetrics, StageMetrics
from ..models.failure_history import FailureEvent, FailureHistory
from ..models.fallback_history import FallbackActivation, FallbackHistory
from ..models.recovery_state import RecoveryAction, RecoveryState
from ..models.timeout_state import TimeoutState
from .base_serializer import BaseSerializer
from .codec import decode_datetime, decode_enum, decode_uuid, encode

# ─── helpers ─────────────────────────────────────────────────────────────────

def _meta_from(d: dict[str, Any]) -> StateMetadata:
    raw = d.get("metadata") or {}
    return StateMetadata(
        created_at=decode_datetime(raw.get("created_at")),
        updated_at=decode_datetime(raw.get("updated_at")),
        producing_node=raw.get("producing_node"),
        checkpoint_id=raw.get("checkpoint_id"),
        schema_version=raw.get("schema_version", 1),
    )


def _ver(d: dict[str, Any]) -> StateVersion:
    return StateVersion(value=d.get("version", 0))


# ─── RecoveryState ────────────────────────────────────────────────────────────

class RecoveryStateSerializer(BaseSerializer[RecoveryState]):
    def serialise(self, s: RecoveryState) -> dict[str, Any]:
        return {
            "investigation_id": str(s.investigation_id),
            "version": s.version.value,
            "recovery_actions": [
                {
                    "action_id": str(a.action_id),
                    "node_name": a.node_name,
                    "action_type": a.action_type,
                    "trigger_failure_code": a.trigger_failure_code,
                    "trigger_failure_type": a.trigger_failure_type,
                    "attempted_at": encode(a.attempted_at),
                    "succeeded": a.succeeded,
                    "note": a.note,
                }
                for a in s.recovery_actions
            ],
            "hitl_escalated_for_failure": s.hitl_escalated_for_failure,
            "hitl_escalation_node": s.hitl_escalation_node,
            "hitl_escalation_reason": s.hitl_escalation_reason,
            "baseline_ref_self_healed": s.baseline_ref_self_healed,
            "baseline_ref_heal_strategy": s.baseline_ref_heal_strategy,
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> RecoveryState:
        actions = tuple(
            RecoveryAction(
                action_id=decode_uuid(a["action_id"]),
                node_name=a.get("node_name", ""),
                action_type=a.get("action_type", ""),
                trigger_failure_code=a.get("trigger_failure_code"),
                trigger_failure_type=a.get("trigger_failure_type"),
                attempted_at=decode_datetime(a.get("attempted_at")),
                succeeded=a.get("succeeded", False),
                note=a.get("note"),
            )
            for a in d.get("recovery_actions", [])
        )
        return RecoveryState(
            version=_ver(d),
            metadata=_meta_from(d),
            investigation_id=decode_uuid(d["investigation_id"]),
            recovery_actions=actions,
            hitl_escalated_for_failure=d.get("hitl_escalated_for_failure", False),
            hitl_escalation_node=d.get("hitl_escalation_node"),
            hitl_escalation_reason=d.get("hitl_escalation_reason"),
            baseline_ref_self_healed=d.get("baseline_ref_self_healed", False),
            baseline_ref_heal_strategy=d.get("baseline_ref_heal_strategy"),
        )


# ─── TimeoutState ─────────────────────────────────────────────────────────────

class TimeoutStateSerializer(BaseSerializer[TimeoutState]):
    def serialise(self, s: TimeoutState) -> dict[str, Any]:
        intervals = [
            [encode(start), encode(end)]
            for start, end in s.hitl_pause_intervals
        ]
        return {
            "investigation_id": str(s.investigation_id),
            "version": s.version.value,
            "started_at": encode(s.started_at),
            "global_budget_seconds": s.global_budget_seconds,
            "hitl_pause_intervals": intervals,
            "global_timeout_triggered": s.global_timeout_triggered,
            "global_timeout_triggered_at": encode(s.global_timeout_triggered_at),
            "node_timeout_events": dict(s.node_timeout_events),
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> TimeoutState:
        intervals = tuple(
            (decode_datetime(pair[0]), decode_datetime(pair[1]))
            for pair in d.get("hitl_pause_intervals", [])
        )
        return TimeoutState(
            version=_ver(d),
            metadata=_meta_from(d),
            investigation_id=decode_uuid(d["investigation_id"]),
            started_at=decode_datetime(d.get("started_at")),
            global_budget_seconds=d.get("global_budget_seconds", 1800),
            hitl_pause_intervals=intervals,
            global_timeout_triggered=d.get("global_timeout_triggered", False),
            global_timeout_triggered_at=decode_datetime(d.get("global_timeout_triggered_at")),
            node_timeout_events=d.get("node_timeout_events", {}),
        )


# ─── FailureHistory ───────────────────────────────────────────────────────────

class FailureHistorySerializer(BaseSerializer[FailureHistory]):
    def serialise(self, s: FailureHistory) -> dict[str, Any]:
        return {
            "investigation_id": str(s.investigation_id),
            "version": s.version.value,
            "events": [
                {
                    "event_id": str(e.event_id),
                    "investigation_id": str(e.investigation_id),
                    "failure_type": e.failure_type.value,
                    "node_name": e.node_name,
                    "stage_key": e.stage_key,
                    "tool_name": e.tool_name,
                    "error_code": e.error_code,
                    "error_detail": e.error_detail,
                    "occurred_at": encode(e.occurred_at),
                    "retry_attempt_at_failure": e.retry_attempt_at_failure,
                    "resolution": e.resolution,
                }
                for e in s.events
            ],
            "counts_by_type": dict(s.counts_by_type),
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> FailureHistory:
        events = tuple(
            FailureEvent(
                event_id=decode_uuid(e["event_id"]),
                investigation_id=decode_uuid(e["investigation_id"]),
                failure_type=decode_enum(FailureType, e.get("failure_type", "schema_violation")),
                node_name=e.get("node_name", ""),
                stage_key=e.get("stage_key", ""),
                tool_name=e.get("tool_name"),
                error_code=e.get("error_code"),
                error_detail=e.get("error_detail"),
                occurred_at=decode_datetime(e.get("occurred_at")),
                retry_attempt_at_failure=e.get("retry_attempt_at_failure", 0),
                resolution=e.get("resolution", "retried"),
            )
            for e in d.get("events", [])
        )
        return FailureHistory(
            version=_ver(d),
            metadata=_meta_from(d),
            investigation_id=decode_uuid(d["investigation_id"]),
            events=events,
            counts_by_type=d.get("counts_by_type", {}),
        )


# ─── FallbackHistory ──────────────────────────────────────────────────────────

class FallbackHistorySerializer(BaseSerializer[FallbackHistory]):
    def serialise(self, s: FallbackHistory) -> dict[str, Any]:
        return {
            "investigation_id": str(s.investigation_id),
            "version": s.version.value,
            "activations": [
                {
                    "activation_id": str(a.activation_id),
                    "investigation_id": str(a.investigation_id),
                    "fallback_type": a.fallback_type,
                    "trigger_criteria": list(a.trigger_criteria),
                    "trace_quality_score_at_trigger": a.trace_quality_score_at_trigger,
                    "trace_coverage_pct_at_trigger": a.trace_coverage_pct_at_trigger,
                    "alternative_stage": a.alternative_stage,
                    "alternative_path_label": a.alternative_path_label,
                    "activated_at": encode(a.activated_at),
                    "fallback_outcome": a.fallback_outcome,
                    "fallback_confidence_degradation": a.fallback_confidence_degradation,
                    "fallback_metrics": a.fallback_metrics,
                }
                for a in s.activations
            ],
            "trace_degradation_count": s.trace_degradation_count,
            "forced_exit_count": s.forced_exit_count,
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> FallbackHistory:
        activations = tuple(
            FallbackActivation(
                activation_id=decode_uuid(a["activation_id"]),
                investigation_id=decode_uuid(a["investigation_id"]),
                fallback_type=a.get("fallback_type", "trace_degradation"),
                trigger_criteria=tuple(a.get("trigger_criteria", [])),
                trace_quality_score_at_trigger=a.get("trace_quality_score_at_trigger"),
                trace_coverage_pct_at_trigger=a.get("trace_coverage_pct_at_trigger"),
                alternative_stage=a.get("alternative_stage", "5.1b"),
                alternative_path_label=a.get("alternative_path_label", ""),
                activated_at=decode_datetime(a.get("activated_at")),
                fallback_outcome=a.get("fallback_outcome", "completed"),
                fallback_confidence_degradation=a.get("fallback_confidence_degradation"),
                fallback_metrics=a.get("fallback_metrics", {}),
            )
            for a in d.get("activations", [])
        )
        return FallbackHistory(
            version=_ver(d),
            metadata=_meta_from(d),
            investigation_id=decode_uuid(d["investigation_id"]),
            activations=activations,
            trace_degradation_count=d.get("trace_degradation_count", 0),
            forced_exit_count=d.get("forced_exit_count", 0),
        )


# ─── ExecutionMetrics ─────────────────────────────────────────────────────────

class ExecutionMetricsSerializer(BaseSerializer[ExecutionMetrics]):
    def serialise(self, s: ExecutionMetrics) -> dict[str, Any]:
        return {
            "investigation_id": str(s.investigation_id),
            "version": s.version.value,
            "total_active_seconds": s.total_active_seconds,
            "total_hitl_pause_seconds": s.total_hitl_pause_seconds,
            "total_token_spend": s.total_token_spend,
            "total_tool_calls": s.total_tool_calls,
            "successful_tool_calls": s.successful_tool_calls,
            "failed_tool_calls": s.failed_tool_calls,
            "circuit_breaker_rejections": s.circuit_breaker_rejections,
            "total_retries": s.total_retries,
            "total_schema_repairs": s.total_schema_repairs,
            "total_hitl_escalations": s.total_hitl_escalations,
            "trace_degradation_active": s.trace_degradation_active,
            "forced_exit_occurred": s.forced_exit_occurred,
            "stage_metrics": [
                {
                    "stage_key": m.stage_key,
                    "tool_calls": m.tool_calls,
                    "successful_tool_calls": m.successful_tool_calls,
                    "failed_tool_calls": m.failed_tool_calls,
                    "retry_count": m.retry_count,
                    "token_spend": m.token_spend,
                    "elapsed_s": m.elapsed_s,
                }
                for m in s.stage_metrics
            ],
            "final_confidence_level": s.final_confidence_level,
            "computed_at": encode(s.computed_at),
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> ExecutionMetrics:
        stage_metrics = tuple(
            StageMetrics(
                stage_key=m.get("stage_key", ""),
                tool_calls=m.get("tool_calls", 0),
                successful_tool_calls=m.get("successful_tool_calls", 0),
                failed_tool_calls=m.get("failed_tool_calls", 0),
                retry_count=m.get("retry_count", 0),
                token_spend=m.get("token_spend", 0),
                elapsed_s=m.get("elapsed_s"),
            )
            for m in d.get("stage_metrics", [])
        )
        return ExecutionMetrics(
            version=_ver(d),
            metadata=_meta_from(d),
            investigation_id=decode_uuid(d["investigation_id"]),
            total_active_seconds=d.get("total_active_seconds"),
            total_hitl_pause_seconds=d.get("total_hitl_pause_seconds"),
            total_token_spend=d.get("total_token_spend", 0),
            total_tool_calls=d.get("total_tool_calls", 0),
            successful_tool_calls=d.get("successful_tool_calls", 0),
            failed_tool_calls=d.get("failed_tool_calls", 0),
            circuit_breaker_rejections=d.get("circuit_breaker_rejections", 0),
            total_retries=d.get("total_retries", 0),
            total_schema_repairs=d.get("total_schema_repairs", 0),
            total_hitl_escalations=d.get("total_hitl_escalations", 0),
            trace_degradation_active=d.get("trace_degradation_active", False),
            forced_exit_occurred=d.get("forced_exit_occurred", False),
            stage_metrics=stage_metrics,
            final_confidence_level=d.get("final_confidence_level"),
            computed_at=decode_datetime(d.get("computed_at")),
        )


# ─── CheckpointState ──────────────────────────────────────────────────────────

class CheckpointStateSerializer(BaseSerializer[CheckpointState]):
    """Serializes CheckpointState metadata row (not the payload column)."""

    def serialise(self, s: CheckpointState) -> dict[str, Any]:
        return {
            "checkpoint_id": str(s.checkpoint_id),
            "thread_id": str(s.thread_id),
            "parent_checkpoint_id": str(s.parent_checkpoint_id) if s.parent_checkpoint_id else None,
            "producing_node": s.producing_node,
            "sequence_number": s.sequence_number,
            "created_at": encode(s.created_at),
            "payload_schema_version": s.payload_schema_version,
            "is_latest": s.is_latest,
            "checkpoint_status": s.checkpoint_status,
            "channel_versions": s.channel_versions,
            "pending_writes": s.pending_writes,
            "version": s.version.value,
            "metadata": encode(s.metadata),
        }

    def deserialise(self, d: dict[str, Any]) -> CheckpointState:
        return CheckpointState(
            version=_ver(d),
            metadata=_meta_from(d),
            checkpoint_id=decode_uuid(d["checkpoint_id"]),
            thread_id=decode_uuid(d["thread_id"]),
            parent_checkpoint_id=decode_uuid(d.get("parent_checkpoint_id")),
            producing_node=d.get("producing_node", ""),
            sequence_number=d.get("sequence_number", 0),
            created_at=decode_datetime(d.get("created_at")),
            payload_schema_version=d.get("payload_schema_version", 1),
            is_latest=d.get("is_latest", True),
            checkpoint_status=d.get("checkpoint_status", "active"),
            channel_versions=d.get("channel_versions", {}),
            pending_writes=d.get("pending_writes", []),
        )
