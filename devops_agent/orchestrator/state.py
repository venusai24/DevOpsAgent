"""LangGraph InvestigationState definition based on ADR-002 Section 6."""

from datetime import datetime
from typing import Any, Literal, TypedDict


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
    investigation_id: str
    time_range: tuple[datetime, datetime]
    explicit_symptoms: dict[str, Any]
    
    # ── Data Sources ──────────────────────────────────────────────────
    app_stats_path: str
    metrics_path: str
    logs_path: str
    traces_path: str
    
    # ── Stage -1: Deduplication ───────────────────────────────────────
    dedup_decision: Literal["NEW", "DUPLICATE", "SUBSET", "SUPERSET", "PARTIAL_OVERLAP"]
    reuse_investigation_id: str | None
    stage0_artifacts_available: bool
    concurrent_investigation_ids: list[str]
    
    # ── Stage 0: Context Assembly ─────────────────────────────────────
    component_registry: dict[str, ComponentInfo]
    declared_topology_graph: dict[str, list[str]]
    tc_to_operation_map: dict[str, dict[str, Any]]
    stack_kpi_map: list[tuple[tuple[str, str], list[str]]] # Serialized tuple keys
    baseline_registry_ref: str
    stage_0_gaps: list[dict[str, Any]]
    
    # ── Stages 1–4: Triage ────────────────────────────────────────────
    blast_radius: Literal["localized", "selective", "broad", "systemic"]
    blast_radius_qualifier: Literal["simultaneous", "sequential"]
    symptom_pattern: str
    anomalous_tc_values: dict[str, dict[str, Any]]
    affected_component_candidates: list[str]
    T0: datetime
    T0_sources: dict[str, datetime | None]
    leading_indicators: dict[str, dict[str, Any]]
    incident_state: Literal["ongoing", "resolved"]
    investigation_cluster: list[str]
    concurrent_incident_clusters: list[list[str]]
    boundary_ambiguous_components: list[str]
    ranked_hypotheses: list[HypothesisSpec]
    
    # ── Stages 5–8: RCA ───────────────────────────────────────────────
    evidence_matrix: dict[str, list[EvidenceItem]]
    updated_hypothesis_scores: dict[str, float]
    eliminated_hypotheses: list[dict[str, Any]]
    surviving_hypotheses: list[str]
    primary_bottleneck: dict[str, Any] | None
    refined_dependency_graph: dict[str, list[dict[str, Any]]]
    undeclared_dependencies: list[dict[str, Any]]
    propagation_verified_pairs: list[dict[str, Any]]
    root_cause_candidate: dict[str, Any] | None
    causal_chain: list[CausalChainHop]
    confidence_level: Literal["HIGH", "MEDIUM", "LOW", "INCONCLUSIVE"] | None
    unconfirmed_links: list[dict[str, Any]]
    final_report: dict[str, Any] | None
    
    # ── Cross-cutting ─────────────────────────────────────────────────
    investigation_state: Literal["active", "AMBIGUOUS_PRE_EVIDENCE", "AMBIGUOUS", "INCONCLUSIVE", "complete"]
    investigation_gaps: list[dict[str, Any]]
    hitl_requests: list[dict[str, Any]]
    hitl_responses: list[dict[str, Any]]
    hitl_resume_action: Literal["RESTART_TRIAGE", "RESTART_EVIDENCE", "RESTART_REASONING", "RESTART_CONTEXT", "FORCE_CLOSE"] | None
    current_node: str
    error_log: list[dict[str, Any]]
    token_spend: dict[str, int]
    wall_clock_seconds: dict[str, float]
