"""Serializer / deserializer for InvestigationState.

Round-trip path:
  InvestigationState  ──serialise──►  dict (JSONB-safe)
                      ◄──deserialise── dict (loaded from Postgres JSONB)

Rules from Phase 3 §7.4:
- All datetime fields → ISO 8601 UTC string.
- UUID fields → lowercase UUID string.
- tuple fields → JSON array (decoded back to tuple on load).
- Enum fields → enum .value string.
- Nested frozen dataclasses → recursive serialisation.
"""

from __future__ import annotations

from typing import Any

from ..enums.confidence_level import ConfidenceLevel
from ..enums.dedup_decision import DedupDecision
from ..enums.investigation_status import InvestigationStatus
from ..models.investigation_state import ExplicitSymptoms, InvestigationState
from .base_serializer import BaseSerializer
from .checkpoint_serializer import (
    ExecutionMetricsSerializer,
    FailureHistorySerializer,
    FallbackHistorySerializer,
    RecoveryStateSerializer,
    TimeoutStateSerializer,
)
from .codec import decode_datetime, decode_enum, decode_uuid, encode
from .guardrail_serializer import GuardrailStateSerializer

_guardrail_ser = GuardrailStateSerializer()
_recovery_ser = RecoveryStateSerializer()
_timeout_ser = TimeoutStateSerializer()
_failure_ser = FailureHistorySerializer()
_fallback_ser = FallbackHistorySerializer()
_metrics_ser = ExecutionMetricsSerializer()


class InvestigationSerializer(BaseSerializer[InvestigationState]):
    """Full round-trip serializer for InvestigationState.

    Called exclusively by the persistence layer (checkpoint repository and
    langgraph_checkpointer adapter).  Never called by agents or tools.
    """

    SCHEMA_VERSION = 1

    def serialise(self, state: InvestigationState) -> dict[str, Any]:
        return {
            "_schema_version": self.SCHEMA_VERSION,
            "investigation_id": str(state.investigation_id),
            "cluster_id": state.cluster_id,
            "version": state.version.value,
            "status": state.status.value,
            # Stage -1
            "explicit_symptoms": encode(state.explicit_symptoms),
            "dedup_decision": state.dedup_decision.value if state.dedup_decision else None,
            "reuse_investigation_id": str(state.reuse_investigation_id) if state.reuse_investigation_id else None,
            "concurrent_investigation_ids": [str(x) for x in state.concurrent_investigation_ids],
            "stage0_artifacts_available": state.stage0_artifacts_available,
            "stage0_cache_key": state.stage0_cache_key,
            "topology_manifest_version": state.topology_manifest_version,
            # Stage 0
            "declared_topology_graph": state.declared_topology_graph,
            "component_registry": state.component_registry,
            "tc_to_operation_map": state.tc_to_operation_map,
            "stack_kpi_map": state.stack_kpi_map,
            "baseline_registry_ref": state.baseline_registry_ref,
            # Triage
            "investigation_cluster": list(state.investigation_cluster),
            "active_tcs": list(state.active_tcs),
            "triage_summary": state.triage_summary,
            "risk_score": state.risk_score,
            # RCA
            "hypotheses": list(state.hypotheses),
            "surviving_hypotheses": list(state.surviving_hypotheses),
            "best_hypothesis": state.best_hypothesis,
            "trace_degradation_active": state.trace_degradation_active,
            "metric_fallback_context": state.metric_fallback_context,
            # Stage 6
            "refined_dependency_graph": state.refined_dependency_graph,
            "undeclared_dependencies": list(state.undeclared_dependencies),
            # Stage 7
            "confidence_level": state.confidence_level.value if state.confidence_level else None,
            "confidence_rationale": state.confidence_rationale,
            # Stage 8
            "final_report": state.final_report,
            "report_delivered_at": encode(state.report_delivered_at),
            # HITL
            "hitl_escalation_payload": state.hitl_escalation_payload,
            "hitl_response": state.hitl_response,
            # Audit
            "investigation_gaps": list(state.investigation_gaps),
            # Phase 4
            "guardrails": _guardrail_ser.serialise_or_none(state.guardrails),
            "recovery": _recovery_ser.serialise_or_none(state.recovery),
            "timeouts": _timeout_ser.serialise_or_none(state.timeouts),
            "failure_history": _failure_ser.serialise_or_none(state.failure_history),
            "fallback_history": _fallback_ser.serialise_or_none(state.fallback_history),
            "metrics": _metrics_ser.serialise_or_none(state.metrics),
            # Base
            "metadata": encode(state.metadata),
        }

    def deserialise(self, data: dict[str, Any]) -> InvestigationState:
        from ..models.base import StateMetadata, StateVersion
        version = StateVersion(value=data.get("version", 0))
        meta_raw = data.get("metadata") or {}
        metadata = StateMetadata(
            created_at=decode_datetime(meta_raw.get("created_at")),
            updated_at=decode_datetime(meta_raw.get("updated_at")),
            producing_node=meta_raw.get("producing_node"),
            checkpoint_id=meta_raw.get("checkpoint_id"),
            schema_version=meta_raw.get("schema_version", 1),
        )

        def _uuids(lst):
            return tuple(decode_uuid(x) for x in (lst or []) if x)

        symptoms_raw = data.get("explicit_symptoms")
        explicit_symptoms = None
        if symptoms_raw:
            explicit_symptoms = ExplicitSymptoms(
                cmdb_ids=tuple(symptoms_raw.get("cmdb_ids", [])),
                tc_values=tuple(symptoms_raw.get("tc_values", [])),
                description=symptoms_raw.get("description", ""),
                alert_timestamp=decode_datetime(symptoms_raw.get("alert_timestamp")),
                raw_alert_payload=symptoms_raw.get("raw_alert_payload", {}),
            )

        guardrails_raw = data.get("guardrails")
        recovery_raw = data.get("recovery")
        timeouts_raw = data.get("timeouts")
        failure_raw = data.get("failure_history")
        fallback_raw = data.get("fallback_history")
        metrics_raw = data.get("metrics")

        return InvestigationState(
            version=version,
            metadata=metadata,
            investigation_id=decode_uuid(data["investigation_id"]),
            cluster_id=data.get("cluster_id", ""),
            status=decode_enum(InvestigationStatus, data.get("status", "claimed")),
            explicit_symptoms=explicit_symptoms,
            dedup_decision=decode_enum(DedupDecision, data.get("dedup_decision")) if data.get("dedup_decision") else None,
            reuse_investigation_id=decode_uuid(data.get("reuse_investigation_id")),
            concurrent_investigation_ids=_uuids(data.get("concurrent_investigation_ids")),
            stage0_artifacts_available=data.get("stage0_artifacts_available", False),
            stage0_cache_key=data.get("stage0_cache_key"),
            topology_manifest_version=data.get("topology_manifest_version"),
            declared_topology_graph=data.get("declared_topology_graph"),
            component_registry=data.get("component_registry"),
            tc_to_operation_map=data.get("tc_to_operation_map"),
            stack_kpi_map=data.get("stack_kpi_map"),
            baseline_registry_ref=data.get("baseline_registry_ref"),
            investigation_cluster=tuple(data.get("investigation_cluster", [])),
            active_tcs=tuple(data.get("active_tcs", [])),
            triage_summary=data.get("triage_summary"),
            risk_score=data.get("risk_score"),
            hypotheses=tuple(data.get("hypotheses", [])),
            surviving_hypotheses=tuple(data.get("surviving_hypotheses", [])),
            best_hypothesis=data.get("best_hypothesis"),
            trace_degradation_active=data.get("trace_degradation_active", False),
            metric_fallback_context=data.get("metric_fallback_context"),
            refined_dependency_graph=data.get("refined_dependency_graph"),
            undeclared_dependencies=tuple(data.get("undeclared_dependencies", [])),
            confidence_level=decode_enum(ConfidenceLevel, data.get("confidence_level")) if data.get("confidence_level") else None,
            confidence_rationale=data.get("confidence_rationale"),
            final_report=data.get("final_report"),
            report_delivered_at=decode_datetime(data.get("report_delivered_at")),
            hitl_escalation_payload=data.get("hitl_escalation_payload"),
            hitl_response=data.get("hitl_response"),
            investigation_gaps=tuple(data.get("investigation_gaps", [])),
            guardrails=_guardrail_ser.deserialise(guardrails_raw) if guardrails_raw else None,
            recovery=_recovery_ser.deserialise(recovery_raw) if recovery_raw else None,
            timeouts=_timeout_ser.deserialise(timeouts_raw) if timeouts_raw else None,
            failure_history=_failure_ser.deserialise(failure_raw) if failure_raw else None,
            fallback_history=_fallback_ser.deserialise(fallback_raw) if fallback_raw else None,
            metrics=_metrics_ser.deserialise(metrics_raw) if metrics_raw else None,
        )
