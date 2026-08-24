"""LangGraph InvestigationState definition based on ADR-002 Section 6."""

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, TypeVar, TypedDict

from langchain_core.messages import AnyMessage

_T = TypeVar("_T")

def _keep_last(a: _T, b: _T) -> _T:  # noqa: ARG001
    """Reducer that always keeps the most recent (last-writer-wins) value.

    Used for singleton state fields that may be written concurrently by
    parallel RCA fan-out branches.  All branches receive the same value
    from dispatch and write it back unchanged, so last-writer-wins is safe.
    """
    return b


class ComponentInfo(TypedDict):
    name: str
    role: Literal["gateway", "app_server", "database", "cache", "container"]
    layer: str
    stack_type: str
    role_source: Literal["cmdb_confirmed", "telemetry_override", "cmdb_only"]
    cmdb_declared_role: str | None

class HypothesisSpec(TypedDict):
    name: str
    prior_score: float
    confirming_kpis: list[str]
    confirming_log_patterns: list[str]
    confirming_trace_pattern: str
    disconfirming_conditions: list[str]

class EvidenceItem(TypedDict):
    source: Literal["traces", "metrics", "logs", "app_stats"]
    component_cmdb: str
    evidence_type: str
    onset_timestamp: datetime | None
    z_score: float | None
    pattern_matched: str | None
    confirming: bool

class CausalChainHop(TypedDict):
    component_name: str
    cmdb_id: str
    failure_mode: str
    onset_T0: datetime
    link_confidence: Literal["confirmed", "unconfirmed"]
    link_type: Literal["trace_verified", "metric_inferred", "declared"]

class InvestigationState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────
    # Annotated with _keep_last so parallel RCA fan-out branches can all
    # write back the same investigation_id without triggering
    # LangGraph's InvalidUpdateError ("Can receive only one value per step").
    investigation_id: Annotated[str, _keep_last]
    time_range: Annotated[tuple[datetime, datetime], _keep_last]
    explicit_symptoms: Annotated[dict[str, Any], _keep_last]
    
    # ── Data Sources ──────────────────────────────────────────────────
    app_stats_path: Annotated[str, _keep_last]
    metrics_path: Annotated[str, _keep_last]
    logs_path: Annotated[str, _keep_last]
    traces_path: Annotated[str, _keep_last]
    
    # ── Stage -1: Deduplication ───────────────────────────────────────
    dedup_decision: Annotated[Literal["NEW", "DUPLICATE", "SUBSET", "SUPERSET", "PARTIAL_OVERLAP"], _keep_last]
    reuse_investigation_id: Annotated[str | None, _keep_last]
    stage0_artifacts_available: Annotated[bool, _keep_last]
    concurrent_investigation_ids: Annotated[list[str], _keep_last]
    
    # ── Stage 0: Context Assembly ─────────────────────────────────────
    component_registry: Annotated[dict[str, ComponentInfo], _keep_last]
    declared_topology_graph: Annotated[dict[str, list[str]], _keep_last]
    discovered_topology_graph: Annotated[dict[str, list[str]], _keep_last]
    tc_to_operation_map: Annotated[dict[str, dict[str, Any]], _keep_last]
    stack_kpi_map: Annotated[list[tuple[tuple[str, str], list[str]]], _keep_last] # Serialized tuple keys
    # Maps each cmdb_id to the exact kpi_name strings present in its baseline.
    # Populated in context_assembly; consumed by the RCA LLM to avoid guessing metric names.
    component_kpi_map: Annotated[dict[str, list[str]], _keep_last]
    baseline_registry_ref: Annotated[str, _keep_last]
    stage_0_gaps: Annotated[list[dict[str, Any]], _keep_last]
    
    # ── Stages 1–4: Triage ────────────────────────────────────────────
    blast_radius: Annotated[Literal["localized", "selective", "broad", "systemic"], _keep_last]
    blast_radius_qualifier: Annotated[Literal["simultaneous", "sequential"], _keep_last]
    symptom_pattern: Annotated[str, _keep_last]
    anomalous_tc_values: Annotated[dict[str, dict[str, Any]], _keep_last]
    affected_component_candidates: Annotated[list[str], _keep_last]
    T0: Annotated[datetime, _keep_last]
    T0_sources: Annotated[dict[str, datetime | None], _keep_last]
    leading_indicators: Annotated[dict[str, dict[str, Any]], _keep_last]
    incident_state: Annotated[Literal["ongoing", "resolved"], _keep_last]
    investigation_cluster: Annotated[list[str], _keep_last]
    concurrent_incident_clusters: Annotated[list[list[str]], _keep_last]
    boundary_ambiguous_components: Annotated[list[str], _keep_last]
    ranked_hypotheses: Annotated[list[HypothesisSpec], _keep_last]
    
    # ── Stages 5–8: RCA ───────────────────────────────────────────────
    evidence_matrix: Annotated[dict[str, list[EvidenceItem]], _keep_last]
    evidence_items: Annotated[list[dict[str, Any]], _keep_last]
    updated_hypothesis_scores: Annotated[dict[str, float], _keep_last]
    eliminated_hypotheses: Annotated[list[dict[str, Any]], _keep_last]
    surviving_hypotheses: Annotated[list[str], _keep_last]
    primary_bottleneck: Annotated[dict[str, Any] | None, _keep_last]
    refined_dependency_graph: Annotated[dict[str, list[dict[str, Any]]], _keep_last]
    undeclared_dependencies: Annotated[list[dict[str, Any]], _keep_last]
    propagation_verified_pairs: Annotated[list[dict[str, Any]], _keep_last]
    root_cause_candidate: Annotated[dict[str, Any] | None, _keep_last]
    causal_chain: Annotated[list[CausalChainHop], _keep_last]
    confidence_level: Annotated[Literal["HIGH", "MEDIUM", "LOW", "INCONCLUSIVE"] | None, _keep_last]
    unconfirmed_links: Annotated[list[dict[str, Any]], _keep_last]
    final_report: Annotated[dict[str, Any] | None, _keep_last]
    evidence_log: Annotated[list[dict[str, Any]], operator.add]
    mismatched_items: Annotated[list[str], _keep_last]
    critic_verdicts: Annotated[list[dict[str, Any]], operator.add]
    current_evidence_items_to_review: Annotated[list[dict[str, Any]], _keep_last]
    # Fan-out accumulator: each parallel RCA branch appends its result dict here.
    # The operator.add reducer ensures branches never clobber each other.
    per_branch_rca_results: Annotated[list[dict[str, Any]], operator.add]
    # Top-level critic feedback injected into every new RCA branch on re-dispatch.
    # Written by critic_agent_node; consumed by rca_dispatch_node.
    critic_feedback_for_rca: str | None
    
    # ── Cross-cutting ─────────────────────────────────────────────────
    investigation_state: Annotated[Literal["active", "AMBIGUOUS_PRE_EVIDENCE", "AMBIGUOUS", "INCONCLUSIVE", "complete"], _keep_last]
    current_trace_run_id: Annotated[str | None, _keep_last]
    investigation_gaps: Annotated[list[dict[str, Any]], _keep_last]
    hitl_requests: Annotated[list[dict[str, Any]], _keep_last]
    hitl_responses: Annotated[list[dict[str, Any]], _keep_last]
    hitl_resume_action: Annotated[Literal["RESTART_TRIAGE", "RESTART_EVIDENCE", "RESTART_REASONING", "RESTART_CONTEXT", "FORCE_CLOSE", "RESTART_RCA"] | None, _keep_last]
    hitl_human_hint: Annotated[str | None, _keep_last]  # Free-text hint from the human supervisor
    current_node: Annotated[str, _keep_last]
    error_log: Annotated[list[dict[str, Any]], _keep_last]
    token_spend: Annotated[dict[str, int], _keep_last]
    wall_clock_seconds: Annotated[dict[str, float], _keep_last]
    rca_messages: Annotated[list[AnyMessage], operator.add]
    turn_count: Annotated[int, _keep_last]
    model_used: Annotated[str, _keep_last]
    verification_failures: Annotated[int, _keep_last]
