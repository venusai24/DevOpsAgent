"""
Investigation Graph and State Models.

InvestigationGraph is the directed evidence graph accumulating across hops.
InvestigationState is the single top-level container passed through the
entire Temporal workflow — the source of truth for investigation progress.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from airs.models.context import ContextBudget, ContextMetrics, InsightTiers
from airs.models.edges import DirectedEdge
from airs.models.evidence import EvidenceNode
from airs.models.hypothesis import Hypothesis
from airs.models.pursuit import InformationPursuitState
from airs.models.risk import TrajectoryRiskState


# ─── Investigation Graph ──────────────────────────────────────────────────────

class InvestigationGraph(BaseModel):
    """
    Directed acyclic graph of evidence nodes connected by typed edges.

    nodes:            Dict[node_id, EvidenceNode] for O(1) lookup.
    edges:            List of all directed edges in the graph.
    root_node_id:     The triggering alert node that started the investigation.
    causal_chain_ids: Ordered list of node_ids on the confirmed causal path.
    """
    nodes: dict[str, EvidenceNode] = Field(default_factory=dict)
    edges: list[DirectedEdge] = Field(default_factory=list)
    root_node_id: Optional[str] = None
    causal_chain_ids: list[str] = Field(default_factory=list)

    def add_node(self, node: EvidenceNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: DirectedEdge) -> None:
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[EvidenceNode]:
        return self.nodes.get(node_id)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def causal_chain_length(self) -> int:
        return len(self.causal_chain_ids)

    def get_causal_chain(self) -> list[EvidenceNode]:
        """Return ordered EvidenceNodes on the causal chain."""
        return [
            self.nodes[nid]
            for nid in self.causal_chain_ids
            if nid in self.nodes
        ]

    def get_outbound_edges(self, node_id: str) -> list[DirectedEdge]:
        return [e for e in self.edges if e.source_node_id == node_id]

    def get_inbound_edges(self, node_id: str) -> list[DirectedEdge]:
        return [e for e in self.edges if e.target_node_id == node_id]


# ─── Alert Payload ────────────────────────────────────────────────────────────

class AlertPayload(BaseModel):
    """
    The triggering alert that initiates an investigation workflow.
    Ingested from Datadog, Prometheus Alertmanager, or synthetic triggers.
    """
    alert_id: str
    alert_name: str
    severity: str                          # critical / high / medium / low
    service: Optional[str] = None          # Affected service (if known)
    namespace: Optional[str] = None        # K8s namespace (if applicable)
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    fired_at: datetime
    source: str = "unknown"               # alertmanager / datadog / synthetic


# ─── Top-Level Investigation State ────────────────────────────────────────────

class InvestigationState(BaseModel):
    """
    The single authoritative state object threaded through the entire workflow.

    This is serialized to JSON for Temporal activity inputs/outputs.
    All mutations return a new state (treat as immutable in activities).

    Fields:
        investigation_id:    Unique investigation identifier.
        workflow_run_id:     Temporal workflow run ID.
        alert:               The triggering alert.
        total_hop_count:     Number of investigation hops completed.
        graph:               The accumulating Investigation Graph.
        risk_state:          Full conformal risk tracking state.
        pursuit_state:       Information Pursuit (missing mass) tracking.
        context_budget:      Token partition configuration.
        insight_tiers:       Four-tier memory store.
        context_metrics:     Runtime context utilization metrics.
        hypotheses:          Active hypothesis set.
        leading_hypothesis_id: ID of the current leading hypothesis.
        created_at:          Workflow start time.
        last_updated_at:     Time of last state mutation.
    """
    investigation_id: str
    workflow_run_id: str
    alert: AlertPayload

    # Investigation progress
    total_hop_count: int = Field(default=0, ge=0)
    graph: InvestigationGraph = Field(default_factory=InvestigationGraph)

    # Risk & pursuit
    risk_state: TrajectoryRiskState = Field(default_factory=TrajectoryRiskState)
    pursuit_state: InformationPursuitState = Field(
        default_factory=InformationPursuitState
    )

    # Context management
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    insight_tiers: InsightTiers = Field(default_factory=InsightTiers)
    context_metrics: ContextMetrics = Field(default_factory=ContextMetrics)

    # Hypothesis tracking
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    leading_hypothesis_id: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def increment_hop(self) -> "InvestigationState":
        """Return a new state with hop count incremented."""
        return self.model_copy(
            update={
                "total_hop_count": self.total_hop_count + 1,
                "last_updated_at": datetime.now(timezone.utc),
            }
        )

    def get_leading_hypothesis(self) -> Optional[Hypothesis]:
        if self.leading_hypothesis_id is None:
            return None
        return next(
            (h for h in self.hypotheses if h.hypothesis_id == self.leading_hypothesis_id),
            None,
        )

    def get_active_hypotheses(self) -> list[Hypothesis]:
        from airs.models.hypothesis import HypothesisStatus
        return [
            h for h in self.hypotheses
            if h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.LEADING)
        ]
