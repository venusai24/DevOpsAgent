"""
airs_v2/reasoning/hypothesis_engine.py
========================================

Hypothesis Engine — Template-to-Node Candidate Generator
----------------------------------------------------------

The HypothesisEngine bridges Stage 1 (PerceptionReport) and Stage 3
(SymbolicValidator) by generating a set of raw ``CausalHypothesis`` objects
for every classified log event in the report.

Pipeline
--------
1. For each ``RouterResult`` in the ``PerceptionReport``:
   a. Look up the node-type hints for the template key.
   b. Collect all nodes in the blast radius that match the type hints.
   c. Score each candidate: ``base_score = result.confidence × tier_weight``
   d. Apply correlation boost: ``+CORRELATION_BOOST`` if the node appears in
      the historical co-failure partners list.
   e. Build the ``causal_path``: BFS from focal_service → candidate in the
      directed dependency graph, then reversed to get root→symptom order.
   f. Emit a ``CausalHypothesis`` per unique candidate node.

2. De-duplicate by candidate node — keep the highest-scoring hypothesis per
   node (the same node can surface from multiple log events).

Scoring weights
---------------
::

    TIER_WEIGHTS = {"L1": 1.00, "L2": 0.85, "L3": 0.50}
    CORRELATION_BOOST = 0.15

The L3 weight (0.50) reflects that novel patterns have higher uncertainty
— the LLM stub may be imprecise about pattern identity.  However, the
SymbolicValidator still enforces all 5 rules regardless of confidence,
so even a high-confidence L3 pattern cannot escape topological pruning.

Node-type hint mapping
-----------------------
Each template key maps to a list of preferred node types.  Node type
is drawn from ``NodeSnapshot.node_type`` (``"database"``, ``"cache"``,
``"service"``, ``"external"``).

The special hint ``"any"`` means all nodes in the blast radius are
eligible — used for network/infrastructure-level patterns and novel L3
patterns where the scope of impact is unknown.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from airs_v2.context.graph import ContextGraph, NodeSnapshot
from airs_v2.perception.router import PerceptionReport, RouterResult, Tier
from airs_v2.reasoning.causal_types import CausalEvidence, CausalHypothesis
from airs_v2.reasoning.symbolic_validator import TopologyContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

TIER_WEIGHTS: dict[str, float] = {
    "L1": 1.00,
    "L2": 0.85,
    "L3": 0.50,
}

CORRELATION_BOOST: float = 0.15


# ---------------------------------------------------------------------------
# Template → node-type mapping
# ---------------------------------------------------------------------------

#: Maps an L1 template key (or ``"__default__"`` for L3 novel) to a list of
#: node-type hints.  ``"any"`` selects all nodes in the blast radius.
TEMPLATE_NODE_HINTS: dict[str, list[str]] = {
    "connection_pool_exhausted":    ["database"],
    "oom_killed":                   ["cache", "service"],
    "redis_oom_eviction":           ["cache"],
    "http_upstream_unavailable":    ["service", "external"],
    "grpc_deadline_exceeded":       ["service"],
    "network_partition":            ["any"],
    "pod_crash_loop":               ["service"],
    "tls_cert_expired":             ["external", "service"],
    "dns_resolution_failure":       ["external"],
    "database_query_timeout":       ["database"],
    "kafka_consumer_lag":           ["service"],
    "transaction_leak":             ["database"],
    "thread_pool_exhausted":        ["service"],
    "upstream_rate_limited":        ["external", "service"],
    "disk_space_exhausted":         ["database", "service"],
    # All novel / L3 patterns → widest net
    "__default__":                  ["any"],
}


# ---------------------------------------------------------------------------
# Hypothesis Engine
# ---------------------------------------------------------------------------


class HypothesisEngine:
    """
    Generates raw ``CausalHypothesis`` candidates from a ``PerceptionReport``.

    Parameters
    ----------
    context:
        The ``TopologyContext`` for the current incident.  Contains the graph,
        blast radius set, and historical correlation data.
    correlation_partners:
        Set of service names from ``CorrelationRecord.partner_service`` for the
        focal service.  Used to boost hypothesis scores.
    """

    def __init__(
        self,
        context: TopologyContext,
        correlation_partners: Optional[set[str]] = None,
    ) -> None:
        self._ctx = context
        self._partners: set[str] = correlation_partners or set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, report: PerceptionReport) -> list[CausalHypothesis]:
        """
        Generate de-duplicated ``CausalHypothesis`` candidates from a report.

        Returns a flat list sorted by ``raw_score`` descending.  Each candidate
        node appears at most once (highest score retained).

        Results marked ``suppressed_transient=True`` by the ``TransientFilter``
        are skipped entirely — they do not contribute to any hypothesis.  The
        suppressed results remain in ``PerceptionReport.results`` for audit.
        """
        # node → best hypothesis so far
        best: dict[str, CausalHypothesis] = {}

        for result in report.results:
            # Skip transient results that did not reach the recurrence threshold
            if result.suppressed_transient:
                logger.debug(
                    "[HypothesisEngine] Skipping suppressed transient result: "
                    "template=%s log=%s",
                    result.template_key,
                    result.log_entry[:60],
                )
                continue

            candidates = self._candidates_for_result(result)
            for h in candidates:
                prev = best.get(h.candidate_node)
                if prev is None or h.raw_score > prev.raw_score:
                    best[h.candidate_node] = h

        all_candidates = sorted(best.values(), key=lambda h: h.raw_score, reverse=True)

        logger.info(
            "[HypothesisEngine] focal=%s generated=%d candidates from %d log results "
            "(%d suppressed transient)",
            self._ctx.focal_service,
            len(all_candidates),
            len(report.results),
            sum(1 for r in report.results if r.suppressed_transient),
        )
        return all_candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidates_for_result(
        self, result: RouterResult
    ) -> list[CausalHypothesis]:
        """Generate hypothesis candidates for a single RouterResult."""
        tier_weight = TIER_WEIGHTS.get(result.tier.value, 0.50)
        type_hints = TEMPLATE_NODE_HINTS.get(
            result.template_key, TEMPLATE_NODE_HINTS["__default__"]
        )

        # Collect eligible nodes from blast radius
        eligible = self._eligible_nodes(type_hints)

        # Always include focal service (self-fault hypothesis)
        focal_snap = self._ctx.graph.get_node(self._ctx.focal_service)
        if focal_snap is not None and self._ctx.focal_service not in eligible:
            eligible[self._ctx.focal_service] = focal_snap

        evidence = CausalEvidence(
            log_entry=result.log_entry,
            template_key=result.template_key,
            tier=result.tier.value,
            confidence=result.confidence,
            source_description=result.description,
            is_transient=result.suppressed_transient,  # propagate transience flag
        )

        hypotheses: list[CausalHypothesis] = []
        for node_name, snap in eligible.items():
            score = result.confidence * tier_weight

            # Correlation boost
            if node_name in self._partners:
                score += CORRELATION_BOOST

            # Build causal path
            causal_path = self._build_causal_path(node_name)

            h = CausalHypothesis(
                candidate_node=node_name,
                template_key=result.template_key,
                evidence=[evidence],
                raw_score=round(min(score, 1.0), 4),
                causal_path=causal_path,
                node_type_hint=snap.node_type if snap else "any",
            )
            hypotheses.append(h)

        return hypotheses

    def _eligible_nodes(
        self, type_hints: list[str]
    ) -> dict[str, Optional[NodeSnapshot]]:
        """
        Return blast-radius nodes whose node_type matches the given hints.

        ``"any"`` hint bypasses the type filter.
        """
        use_any = "any" in type_hints
        result: dict[str, Optional[NodeSnapshot]] = {}

        for name in self._ctx.blast_radius_nodes:
            snap = self._ctx.graph.get_node(name)
            if use_any:
                result[name] = snap
            elif snap is not None and snap.node_type in type_hints:
                result[name] = snap

        return result

    def _build_causal_path(self, candidate_node: str) -> list[str]:
        """
        Build the causal propagation path from candidate → focal service.

        Finds the shortest path in the directed dependency graph from
        focal_service → candidate (following dependency edges), then reverses
        it to get the causal propagation direction: candidate → focal.

        Returns ``[candidate_node]`` if candidate IS the focal service (self-fault).
        Returns ``[candidate_node, focal_service]`` if there is a direct edge.
        Returns ``[]`` (empty — will fail R4) if no path exists.
        """
        focal = self._ctx.focal_service
        g = self._ctx.graph._g

        if candidate_node == focal:
            return [focal]

        try:
            # Path in dependency direction: focal → ... → candidate
            dep_path = nx.shortest_path(g, focal, candidate_node)
            # Reverse for causal direction: candidate → ... → focal
            return list(reversed(dep_path))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path found — this hypothesis will fail R4 during validation
            return []
