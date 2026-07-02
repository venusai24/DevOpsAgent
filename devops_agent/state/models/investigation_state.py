"""InvestigationState — the full LangGraph-managed investigation payload.

This is the authoritative runtime state for one investigation.  It is the
dict that lives in the LangGraph checkpoint's channel_values, serialised and
deserialised by InvestigationSerializer.

All fields are ADR-002 / ADR-001 derived.  Guardrail sub-state is nested
as guardrails: GuardrailState.  No business logic belongs here.

Rules for field design:
- Primitive types only at the top level for JSON-safe serialisation.
- Complex sub-graphs use frozen dataclasses in the models package.
- Tuples, not lists, for immutable sequences.
- All datetimes are UTC-aware.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums.confidence_level import ConfidenceLevel
from ..enums.dedup_decision import DedupDecision
from ..enums.investigation_status import InvestigationStatus
from .base import BaseState, _new_uuid
from .execution_metrics import ExecutionMetrics
from .failure_history import FailureHistory
from .fallback_history import FallbackHistory
from .guardrail_state import GuardrailState
from .recovery_state import RecoveryState
from .timeout_state import TimeoutState


@dataclass(frozen=True)
class ExplicitSymptoms(BaseState):
    """Alert-provided symptom payload (Stage -1 input)."""
    cmdb_ids: tuple[str, ...] = ()
    tc_values: tuple[str, ...] = ()
    description: str = ""
    alert_timestamp: datetime | None = None
    raw_alert_payload: dict[str, Any] = field(default_factory=dict)
    dependencies_unknown: bool = False


@dataclass(frozen=True)
class InvestigationState(BaseState):
    """Full investigation state, managed by the LangGraph checkpoint store.

    This model encapsulates ALL data an investigation thread needs across
    all stages.  The persistence layer (InvestigationSerializer) is the only
    code that converts to/from the JSON-safe checkpoint payload.

    Naming convention: fields match ADR-001/ADR-002 exactly so that the
    orchestration layer can reference them without aliasing.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    # Equals LangGraph thread_id.
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)
    cluster_id: str = ""

    # ── Stage -1: Deduplication ───────────────────────────────────────────────
    explicit_symptoms: ExplicitSymptoms | None = None
    dedup_decision: DedupDecision | None = None
    reuse_investigation_id: uuid.UUID | None = None
    concurrent_investigation_ids: tuple[uuid.UUID, ...] = ()
    stage0_artifacts_available: bool = False
    stage0_cache_key: str | None = None
    topology_manifest_version: str | None = None

    # ── Stage 0: Context Assembly ─────────────────────────────────────────────
    declared_topology_graph: dict[str, Any] | None = None
    component_registry: dict[str, Any] | None = None
    tc_to_operation_map: dict[str, Any] | None = None
    stack_kpi_map: dict[str, Any] | None = None
    baseline_registry_ref: str | None = None

    # ── Stage 1–4: Triage (ContextAssembler / TriageAgent) ───────────────────
    investigation_cluster: tuple[str, ...] = ()
    active_tcs: tuple[str, ...] = ()
    triage_summary: str | None = None
    risk_score: float | None = None

    # ── Stage 5: RCA Evidence Collection ─────────────────────────────────────
    hypotheses: tuple[dict[str, Any], ...] = ()
    surviving_hypotheses: tuple[dict[str, Any], ...] = ()
    best_hypothesis: dict[str, Any] | None = None

    # ── Stage 5.1b: Trace Degradation Path ───────────────────────────────────
    trace_degradation_active: bool = False
    metric_fallback_context: dict[str, Any] | None = None

    # ── Stage 6: Dependency Refinement ───────────────────────────────────────
    refined_dependency_graph: dict[str, Any] | None = None
    undeclared_dependencies: tuple[str, ...] = ()

    # ── Stage 7: Confidence Scoring ───────────────────────────────────────────
    confidence_level: ConfidenceLevel | None = None
    confidence_rationale: str | None = None

    # ── Stage 8: Report ───────────────────────────────────────────────────────
    final_report: dict[str, Any] | None = None
    report_delivered_at: datetime | None = None

    # ── HITL ─────────────────────────────────────────────────────────────────
    hitl_escalation_payload: dict[str, Any] | None = None
    hitl_response: dict[str, Any] | None = None

    # ── Audit / gaps ─────────────────────────────────────────────────────────
    investigation_gaps: tuple[str, ...] = ()
    status: InvestigationStatus = InvestigationStatus.CLAIMED

    # ── Phase 4 sub-states ────────────────────────────────────────────────────
    guardrails: GuardrailState | None = None
    recovery: RecoveryState | None = None
    timeouts: TimeoutState | None = None
    failure_history: FailureHistory | None = None
    fallback_history: FallbackHistory | None = None
    metrics: ExecutionMetrics | None = None

    def add_gap(self, gap: str) -> InvestigationState:
        return self.evolve(investigation_gaps=self.investigation_gaps + (gap,))

    def with_status(self, status: InvestigationStatus) -> InvestigationState:
        return self.evolve(status=status)

    def with_guardrails(self, gs: GuardrailState) -> InvestigationState:
        return self.evolve(guardrails=gs)

    def with_timeouts(self, ts: TimeoutState) -> InvestigationState:
        return self.evolve(timeouts=ts)
