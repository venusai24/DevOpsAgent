"""
airs_v2/reasoning/causal_types.py
===================================

Pydantic v2 data models for the Stage 3 Reasoning Layer.

Type hierarchy
--------------
CausalEvidence          — A single log event linking a template to a candidate node.
CausalHypothesis        — A candidate root-cause explanation (pre-validation, raw).
RejectedHypothesis      — A hypothesis pruned by the SymbolicValidator + why.
RootCauseHypothesis     — A validated, ranked root-cause with causal path.
CausalNode              — A node in the sparse causal graph.
CausalEdge              — A directed causal edge in the sparse causal graph.
CausalGraph             — The complete validated sparse causal graph.
IncidentAnalysis        — Final output of ReasoningEngine.analyze_incident().

Directionality convention
--------------------------
All causal_path lists run from ROOT CAUSE → SYMPTOM (focal service).
E.g. ``["payments-db", "payments-service"]`` means payments-db failed and
the failure propagated to payments-service.

In the underlying NetworkX DiGraph this corresponds to traversing edges
in *reverse* direction (edges run A→B = A DEPENDS_ON B, so propagation
goes B→A).

Stale-metrics invariant
------------------------
``stale_metrics_warning`` on ``RootCauseHypothesis`` is always ``True`` in
Stage 2 because ``MetricsSnapshot.stale=True`` is hardcoded until the live
metrics API is wired in Stage 4.  The ``SymbolicValidator`` accounts for this:
R5 is advisory-only and never causes a rejection based on stale data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Leaf types
# ---------------------------------------------------------------------------


class CausalEvidence(BaseModel):
    """
    A single log event that provides evidence for a candidate root-cause node.

    Attributes
    ----------
    log_entry:
        The raw log text that was classified by the perception router.
    template_key:
        The canonical failure pattern key (e.g., ``"connection_pool_exhausted"``).
    tier:
        Which perception tier produced the classification (``"L1"``, ``"L2"``, ``"L3"``).
    confidence:
        Classification confidence from the perception router (0.0 – 1.0).
    source_description:
        Human-readable description of the failure pattern.
    """

    log_entry: str
    template_key: str
    tier: str  # "L1" | "L2" | "L3"
    confidence: float = Field(ge=0.0, le=1.0)
    source_description: str = ""
    is_transient: bool = False  # True when this evidence comes from a transient-candidate
                                # template (propagated from RouterResult.suppressed_transient)


# ---------------------------------------------------------------------------
# Pre-validation types
# ---------------------------------------------------------------------------


class CausalHypothesis(BaseModel):
    """
    A candidate root-cause explanation produced by the HypothesisEngine.

    This object has NOT yet been validated by the SymbolicValidator.
    It may be physically impossible — validation happens separately so that
    rejections are auditable.

    Attributes
    ----------
    candidate_node:
        The infrastructure node hypothesised as the root cause.
    template_key:
        The dominant failure template driving this hypothesis.
    evidence:
        All perception events that contributed to this hypothesis.
    raw_score:
        Pre-validation score: ``confidence × tier_weight + correlation_boost``.
    causal_path:
        Proposed propagation path from candidate to focal service,
        e.g. ``["payments-db", "payments-service"]``.
        Empty list means the candidate IS the focal service (self-fault).
    node_type_hint:
        Expected node type of the candidate (``"database"``, ``"cache"``,
        ``"service"``, ``"external"``, ``"any"``).  Used during validation
        for logging but not enforced as a hard constraint.
    """

    candidate_node: str
    template_key: str
    evidence: list[CausalEvidence] = Field(default_factory=list)
    raw_score: float = Field(default=0.0, ge=0.0)
    causal_path: list[str] = Field(default_factory=list)
    node_type_hint: str = "any"


class RejectedHypothesis(BaseModel):
    """
    A ``CausalHypothesis`` that failed symbolic validation.

    Preserved in the ``IncidentAnalysis`` for audit and explainability.

    Attributes
    ----------
    hypothesis:
        The original (invalid) hypothesis.
    violated_rule:
        Machine-readable rule identifier, e.g. ``"R2_BLAST_RADIUS"``.
    reason:
        Human-readable explanation of why the hypothesis was rejected.
    """

    hypothesis: CausalHypothesis
    violated_rule: str
    reason: str


# ---------------------------------------------------------------------------
# Post-validation types
# ---------------------------------------------------------------------------


class RootCauseHypothesis(BaseModel):
    """
    A validated, ranked root-cause hypothesis.

    Attributes
    ----------
    candidate_node:
        Infrastructure node confirmed as a physically plausible root cause.
    template_key:
        Dominant failure template.
    rank:
        1-based rank among all candidates (rank=1 is the most likely).
    final_score:
        Score after applying R5 advisory adjustments and correlation boosts.
    evidence:
        All perception events backing this candidate.
    causal_path:
        Validated propagation path from candidate to focal service.
    propagation_mechanism:
        How the failure propagated (``"depends_on"`` | ``"cascade"`` |
        ``"correlation"``).
    on_call_contacts:
        On-call team contacts from the topology fixture (may be empty).
    stale_metrics_warning:
        Always ``True`` in Stage 2 — live metrics not yet connected.
    """

    candidate_node: str
    template_key: str
    rank: int = Field(default=0, ge=0)
    final_score: float = Field(default=0.0, ge=0.0)
    evidence: list[CausalEvidence] = Field(default_factory=list)
    causal_path: list[str] = Field(default_factory=list)
    propagation_mechanism: Literal["depends_on", "cascade", "correlation"] = "depends_on"
    on_call_contacts: list[str] = Field(default_factory=list)
    stale_metrics_warning: bool = True


# ---------------------------------------------------------------------------
# Causal graph types
# ---------------------------------------------------------------------------


class CausalNode(BaseModel):
    """
    A node in the validated sparse causal graph.

    Attributes
    ----------
    name:
        Service / resource name.
    role:
        ``"root_cause"`` — the hypothesised origin of the fault.
        ``"propagation"`` — an intermediate node on the causal path.
        ``"symptom"`` — the focal service showing symptoms.
        ``"context"`` — in the blast radius but no direct causal role.
    health_status:
        Current health from the Context Graph (``"healthy"`` | ``"degraded"``
        | ``"critical"`` | ``"unknown"``).
    tier:
        Infrastructure tier from the topology fixture (1 = most critical).
    evidence_count:
        Number of perception events pointing at this node.
    """

    name: str
    role: Literal["root_cause", "propagation", "symptom", "context"]
    health_status: str = "unknown"
    tier: int = Field(default=3, ge=1)
    evidence_count: int = Field(default=0, ge=0)


class CausalEdge(BaseModel):
    """
    A directed causal edge in the sparse causal graph.

    Direction: ``source`` → ``target`` means *source* caused *target* to fail.
    This is the **reverse** of the dependency edge direction (in the Context
    Graph, ``target`` DEPENDS_ON ``source``).

    Attributes
    ----------
    source:
        Upstream node (root or propagation).
    target:
        Downstream node that failed because of source.
    mechanism:
        ``"depends_on"`` — direct dependency edge.
        ``"cascade"`` — multi-hop cascaded failure.
        ``"correlation"`` — historical co-failure (no direct dependency).
    strength:
        Causal strength estimate 0.0 – 1.0 (from raw_score of hypothesis).
    """

    source: str
    target: str
    mechanism: Literal["depends_on", "cascade", "correlation"] = "depends_on"
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class CausalGraph(BaseModel):
    """
    The complete validated sparse causal graph for an incident.

    Contains only nodes and edges within the depth-2 blast radius of the focal
    service that have been confirmed by the SymbolicValidator.
    """

    focal_service: str
    nodes: list[CausalNode] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the causal graph as a markdown report for LLM consumption."""
        root_causes = [n for n in self.nodes if n.role == "root_cause"]
        propagation = [n for n in self.nodes if n.role == "propagation"]
        symptom = [n for n in self.nodes if n.role == "symptom"]

        lines = [
            f"## Causal Graph — `{self.focal_service}`",
            f"**Nodes**: {len(self.nodes)}  |  **Edges**: {len(self.edges)}",
            "",
        ]

        if root_causes:
            lines.append("### Root Cause Candidates")
            for n in root_causes:
                health_icon = {"critical": "🔴", "degraded": "🟡", "healthy": "🟢"}.get(
                    n.health_status, "⚪"
                )
                lines.append(
                    f"- {health_icon} **{n.name}** "
                    f"(tier={n.tier}, health={n.health_status}, "
                    f"evidence={n.evidence_count})"
                )

        if propagation:
            lines.append("\n### Propagation Path")
            for n in propagation:
                lines.append(f"- ↳ {n.name} (tier={n.tier})")

        if symptom:
            lines.append("\n### Symptom Node (Focal Service)")
            for n in symptom:
                lines.append(f"- 🎯 **{n.name}** (tier={n.tier})")

        if self.edges:
            lines.append("\n### Causal Edges")
            for e in self.edges:
                mech_arrow = {"depends_on": "──►", "cascade": "══►", "correlation": "- - ►"}.get(
                    e.mechanism, "──►"
                )
                lines.append(
                    f"- `{e.source}` {mech_arrow} `{e.target}` "
                    f"[{e.mechanism}, strength={e.strength:.2f}]"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level analysis output
# ---------------------------------------------------------------------------


class IncidentAnalysis(BaseModel):
    """
    Final output of ``ReasoningEngine.analyze_incident()``.

    Attributes
    ----------
    focal_service:
        The infrastructure service at the centre of the incident.
    causal_graph:
        The validated sparse causal graph.
    root_cause_candidates:
        Ranked list of valid hypotheses (rank=1 is most likely).
    rejected_hypotheses:
        All hypotheses rejected by the SymbolicValidator, with reasons.
    symbolic_path:
        ``"SYMBOLIC_FAST"`` — all events resolved via L1, no LLM needed.
        ``"CBR_GUIDED"`` — mixed L1/L2, historical correlations used.
        ``"NEURAL_FULL"`` — one or more L3 novel patterns processed.
    overall_confidence:
        Weighted average confidence of the top-3 valid hypotheses.
    analysis_markdown:
        LLM-consumable incident summary combining perception + causal graph.
    analyzed_at:
        ISO-8601 timestamp of analysis.
    """

    focal_service: str
    causal_graph: CausalGraph
    root_cause_candidates: list[RootCauseHypothesis] = Field(default_factory=list)
    rejected_hypotheses: list[RejectedHypothesis] = Field(default_factory=list)
    symbolic_path: Literal["SYMBOLIC_FAST", "CBR_GUIDED", "NEURAL_FULL"] = "SYMBOLIC_FAST"
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_markdown: str = ""
    analyzed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
