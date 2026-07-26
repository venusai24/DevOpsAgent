"""Pydantic Output Schemas for LLM Structured Output Parsing."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TriageAgentOutput(BaseModel):
    blast_radius: Literal["localized", "selective", "broad", "systemic"]
    blast_radius_qualifier: Literal["simultaneous", "sequential"]
    symptom_pattern: str
    anomalous_tc_values: dict[str, dict[str, Any]] = Field(default_factory=dict)
    affected_component_candidates: list[str]
    investigation_cluster: list[str]
    ranked_hypotheses: list[dict[str, Any]]
    T0: datetime | None = None
    T0_sources: dict[str, datetime | None] = Field(default_factory=dict)
    leading_indicators: dict[str, dict[str, Any]] = Field(default_factory=dict)
    incident_state: Literal["ongoing", "resolved"] = "ongoing"
    concurrent_incident_clusters: list[list[str]] = Field(default_factory=list)
    boundary_ambiguous_components: list[str] = Field(default_factory=list)
    dependency_graph: dict[str, Any] = Field(default_factory=dict)
    investigation_state: Literal["active", "AMBIGUOUS_PRE_EVIDENCE", "AMBIGUOUS", "INCONCLUSIVE", "complete"] = "active"
    current_node: str = "triage"

class EvidenceItem(BaseModel):
    hypothesis_id: str
    evidence_source: Literal["metric", "log"]
    raw_reference: dict[str, Any]
    directional_support: Literal["strongly_supports", "weakly_supports", "neutral", "weakly_contradicts", "strongly_contradicts"]
    rationale: str

class SubmitEvidenceReport(BaseModel):
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    primary_hypothesis_id: str | None = None
    narrative_summary: str = ""
    is_ready_to_conclude: bool = False
    
    eliminated_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    surviving_hypotheses: list[str] = Field(default_factory=list)
    primary_bottleneck: dict[str, Any] | None = None
    refined_dependency_graph: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    undeclared_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    propagation_verified_pairs: list[dict[str, Any]] = Field(default_factory=list)
    root_cause_candidate: dict[str, Any] | None = Field(default=None, description="The confirmed root cause.")
    causal_chain: list[dict[str, Any]] = Field(default_factory=list, description="Chain of propagation.")
    unconfirmed_links: list[dict[str, Any]] = Field(default_factory=list)
    final_report: dict[str, Any] | None = Field(default=None, description="Structured final report.")
    investigation_state: Literal["active", "AMBIGUOUS", "INCONCLUSIVE", "complete"] = "active"
    investigation_gaps: list[dict[str, Any]] = Field(default_factory=list)
    current_node: str = "rca"

class CriticVerdict(BaseModel):
    evidence_item_id: str
    verdict: Literal["confirm_original", "reject_with_critique"]
    rationale: str
