"""
Context Scorer — Module 1.7.

Computes the 5-dimensional composite relevance score for each EvidenceNode
to determine its priority in the context window.

Scoring formula:
    composite = w_rel * relevance
              + w_rec * recency
              + w_caus * causal_importance
              + w_uniq * uniqueness
              + w_diag * diagnostic_value

All weights from calibration/defaults.json §context.scorer_weights.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from airs.models.context import ContextScore
from airs.models.evidence import EvidenceNode
from airs.models.investigation import InvestigationGraph

log = logging.getLogger(__name__)


class ContextScorer:
    """
    Computes composite context relevance scores for evidence nodes.

    Stateless: scores depend only on the node, the graph, and current
    time. Can be called at any point without modifying any state.
    """

    def __init__(
        self,
        incident_embedding: Optional[list[float]] = None,
        recency_decay_lambda: float = 0.0002,
        eviction_score_threshold: float = 0.30,
        scorer_weights: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Args:
            incident_embedding:      Embedding of the original alert for
                                     semantic relevance computation.
                                     None → relevance defaults to 0.5.
            recency_decay_lambda:    λ for exp(-λ * seconds_since_observation).
            eviction_score_threshold: Nodes below this score are eviction candidates.
            scorer_weights:          Dict of {dimension: weight}. Defaults to
                                     calibration/defaults.json values.
        """
        self._incident_embedding = incident_embedding
        self._lambda = recency_decay_lambda
        self._eviction_threshold = eviction_score_threshold
        self._weights = scorer_weights or {
            "relevance": 0.30,
            "recency": 0.15,
            "causal_importance": 0.25,
            "uniqueness": 0.15,
            "diagnostic_value": 0.15,
        }

    @classmethod
    def from_calibration(
        cls,
        incident_embedding: Optional[list[float]] = None,
    ) -> "ContextScorer":
        """Create ContextScorer from calibration defaults."""
        from airs.risk.calibration import get_calibration_store
        store = get_calibration_store()
        return cls(
            incident_embedding=incident_embedding,
            recency_decay_lambda=store.get_recency_decay_lambda(),
            eviction_score_threshold=store.get_eviction_score_threshold(),
            scorer_weights=store.get_scorer_weights(),
        )

    def score_node(
        self,
        node: EvidenceNode,
        graph: InvestigationGraph,
        current_hop: int,
    ) -> ContextScore:
        """
        Compute the 5-dimensional ContextScore for an EvidenceNode.

        Args:
            node:        The EvidenceNode to score.
            graph:       Current Investigation Graph (for causal chain lookups).
            current_hop: Current investigation hop (for recency decay reference).

        Returns:
            ContextScore with composite value.
        """
        relevance = self._compute_relevance(node)
        recency = self._compute_recency(node)
        causal_importance = self._compute_causal_importance(node, graph)
        uniqueness = self._compute_uniqueness(node, graph)
        diagnostic_value = self._compute_diagnostic_value(node)

        return ContextScore(
            relevance=relevance,
            recency=recency,
            causal_importance=causal_importance,
            uniqueness=uniqueness,
            diagnostic_value=diagnostic_value,
        )

    def score_all_nodes(
        self,
        nodes: list[EvidenceNode],
        graph: InvestigationGraph,
        current_hop: int,
    ) -> dict[str, ContextScore]:
        """Score all nodes and return {node_id: ContextScore}."""
        return {
            node.node_id: self.score_node(node, graph, current_hop)
            for node in nodes
        }

    def is_eviction_candidate(self, score: ContextScore) -> bool:
        """True if the composite score is below the eviction threshold."""
        return score.composite < self._eviction_threshold

    # ─── Dimension Scorers ────────────────────────────────────────────────────

    def _compute_relevance(self, node: EvidenceNode) -> float:
        """
        Semantic relevance to the incident symptom.

        Phase 1: Uses cosine similarity between node embedding and incident
        embedding if available; otherwise uses a proxy based on the
        node's nonconformity score inversion (1 - s_k).
        """
        if self._incident_embedding is not None:
            # Try to get the node's embedding from the context dict
            node_embedding = node.context.get("_embedding")
            if node_embedding is not None:
                return self._cosine_similarity(
                    self._incident_embedding, node_embedding
                )

        # Proxy: invert the nonconformity score
        # High nonconformity = uncertain/irrelevant → lower relevance
        return max(0.0, 1.0 - node.uncertainty.nonconformity_score)

    def _compute_recency(self, node: EvidenceNode) -> float:
        """
        Exponential time decay: score = exp(-λ * seconds_since_observation).

        Recent evidence is highly relevant; old evidence decays.
        """
        now = datetime.now(timezone.utc)
        observed = node.temporality.observed_at

        # Handle timezone-naive datetimes
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)

        seconds_ago = max(0.0, (now - observed).total_seconds())
        return math.exp(-self._lambda * seconds_ago)

    def _compute_causal_importance(
        self, node: EvidenceNode, graph: InvestigationGraph
    ) -> float:
        """
        Causal importance based on graph centrality.

        Computation:
        - Is on confirmed causal chain → 1.0
        - Is hub node (many outbound edges) → scaled by edge count
        - Otherwise → scaled by BERTScore F1 (proxy for causal grounding)
        """
        if node.is_causal:
            return 1.0

        # Hub node: more outbound edges = more causal
        outbound_edges = graph.get_outbound_edges(node.node_id)
        if outbound_edges:
            # Normalise: cap at 5 outbound edges for max score = 0.90
            hub_score = min(0.90, len(outbound_edges) * 0.18)
            return hub_score

        # Default: use BERTScore F1 as causal grounding proxy
        return node.uncertainty.bert_f1_score * 0.6

    def _compute_uniqueness(
        self, node: EvidenceNode, graph: InvestigationGraph
    ) -> float:
        """
        Uniqueness: 1 - (overlap with existing evidence).

        Computation:
        - Count how many nodes in the graph share the same entity AND signal source
        - More duplicates → lower uniqueness
        """
        entity_guid = str(node.entity.canonical_guid)
        same_entity_same_signal = sum(
            1
            for existing_id, existing_node in graph.nodes.items()
            if existing_id != node.node_id
            and str(existing_node.entity.canonical_guid) == entity_guid
            and existing_node.signal_source == node.signal_source
        )

        # Each duplicate reduces uniqueness by 0.20 (cap at 4 duplicates = 0.0)
        return max(0.0, 1.0 - same_entity_same_signal * 0.20)

    @staticmethod
    def _compute_diagnostic_value(node: EvidenceNode) -> float:
        """
        Diagnostic value = estimated impact on hypothesis confidence.

        Phase 1 proxy: nodes on the causal chain get high value;
        others use BERTScore F1 as a quality proxy.
        """
        if node.is_causal:
            return 0.90

        # BERTScore F1 = semantic fidelity of interpretation → proxy for
        # diagnostic utility (well-grounded interpretations = more useful)
        return node.uncertainty.bert_f1_score

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))
