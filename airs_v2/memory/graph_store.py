"""
airs_v2/memory/graph_store.py
==============================

Abstract GraphStore protocol + NetworkX Phase 1 implementation.

Phase 1 (this file) — NetworkXGraphStore
  - In-process NetworkX DiGraph
  - JSON persistence via nx.readwrite.json_graph (survives restarts)
  - Zero infrastructure: no Docker, no database setup
  - Consistent with the existing ContextGraph (also NetworkX)

Phase 2 (future) — Neo4jGraphStore
  - Swap via config.py GRAPH_BACKEND = "neo4j"
  - Implements the same GraphStore protocol
  - Application code requires zero changes between phases

Graph schema
------------
Nodes:
  (:Incident  {trace_id, incident_type, outcome, confidence, created_at})
  (:Service   {name, tier})
  (:Pattern   {template_key, count})
  (:Runbook   {url})

Edges:
  (:Incident)-[:FOCAL_SERVICE]->(:Service)
  (:Incident)-[:ROOT_CAUSE]->(:Service)
  (:Incident)-[:AFFECTED]->(:Service)
  (:Incident)-[:EXHIBITED]->(:Pattern)
  (:Service)-[:MISDIAGNOSED_AS {count}]->(:Service)   ← from HITL reject feedback
  (:Incident)-[:MODIFIED_BY {reviewer_id}]->(:Incident)
  (:Incident)-[:REFERENCES]->(:Runbook)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphStore(Protocol):
    """Abstract interface for the Knowledge Graph backend."""

    def ingest_trace(self, trace: Any) -> None: ...
    def ingest_feedback(self, trace_id: str, feedback: Any) -> None: ...
    def get_service_incident_history(
        self, service: str, limit: int
    ) -> list[dict]: ...
    def get_pattern_frequency(self, template_key: str) -> dict: ...
    def get_causal_chains(
        self, service: str, depth: int
    ) -> list[list[str]]: ...
    def get_misdiagnosis_history(self, service: str) -> list[dict]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# NetworkX implementation
# ---------------------------------------------------------------------------


class NetworkXGraphStore:
    """
    NetworkX-backed Knowledge Graph with JSON file persistence.

    Every write flushes to a JSON file using node_link_data format,
    so the graph survives process restarts. The flush is synchronous and
    fast for graph sizes expected in local development (< 10,000 nodes).

    Parameters
    ----------
    persist_path : Path to the JSON persistence file.
    """

    def __init__(
        self, persist_path: Optional[str | Path] = None
    ) -> None:
        import networkx as nx
        from config import settings

        self._nx = nx
        self._path = Path(persist_path or settings.GRAPH_PERSIST_PATH)
        self._g: nx.DiGraph = nx.DiGraph()
        self._load()
        logger.info(
            "[NetworkXGraphStore] Loaded graph: %d nodes, %d edges from %s",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
            self._path,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load graph from JSON file if it exists."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._g = self._nx.readwrite.json_graph.node_link_graph(data)
                logger.debug(
                    "[NetworkXGraphStore] Graph loaded from %s", self._path
                )
            except Exception as exc:
                logger.warning(
                    "[NetworkXGraphStore] Could not load graph from %s: %s — starting fresh.",
                    self._path,
                    exc,
                )

    def _flush(self) -> None:
        """Persist current graph state to JSON file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = self._nx.readwrite.json_graph.node_link_data(self._g)
            self._path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("[NetworkXGraphStore] Flush failed: %s", exc)

    # ── Write ──────────────────────────────────────────────────────────────────

    def ingest_trace(self, trace: Any) -> None:
        """
        Ingest a PostMortemTrace into the graph, creating/merging nodes and edges.

        Nodes created:
          (:Incident) (:Service {name=focal_service}) (:Service {affected...})
          (:Pattern {template_key=incident_type})

        Edges created:
          Incident -[FOCAL_SERVICE]-> Service
          Incident -[ROOT_CAUSE]-> Service
          Incident -[AFFECTED]-> Service (for each affected service)
          Incident -[EXHIBITED]-> Pattern
        """
        incident_id = f"incident:{trace.trace_id}"

        # Incident node
        self._g.add_node(
            incident_id,
            node_type="incident",
            trace_id=trace.trace_id,
            incident_type=trace.incident_type,
            outcome=trace.outcome.value if hasattr(trace.outcome, "value") else str(trace.outcome),
            diagnosis_confidence=trace.diagnosis_confidence,
            symbolic_path=trace.symbolic_path,
            created_at=trace.created_at,
        )

        # Focal service node
        focal_id = f"service:{trace.focal_service}"
        if not self._g.has_node(focal_id):
            self._g.add_node(focal_id, node_type="service", name=trace.focal_service)
        self._g.add_edge(incident_id, focal_id, rel="FOCAL_SERVICE")

        # Root cause service node
        if trace.root_cause_node:
            rca_id = f"service:{trace.root_cause_node}"
            if not self._g.has_node(rca_id):
                self._g.add_node(
                    rca_id, node_type="service", name=trace.root_cause_node
                )
            self._g.add_edge(incident_id, rca_id, rel="ROOT_CAUSE")

        # Affected services
        for svc in trace.affected_services:
            svc_id = f"service:{svc}"
            if not self._g.has_node(svc_id):
                self._g.add_node(svc_id, node_type="service", name=svc)
            self._g.add_edge(incident_id, svc_id, rel="AFFECTED")

        # Failure pattern node
        if trace.incident_type:
            pattern_id = f"pattern:{trace.incident_type}"
            if not self._g.has_node(pattern_id):
                self._g.add_node(
                    pattern_id,
                    node_type="pattern",
                    template_key=trace.incident_type,
                    count=0,
                )
            self._g.nodes[pattern_id]["count"] = (
                self._g.nodes[pattern_id].get("count", 0) + 1
            )
            self._g.add_edge(incident_id, pattern_id, rel="EXHIBITED")

        self._flush()
        logger.debug("[NetworkXGraphStore] Ingested trace %s", trace.trace_id)

    def ingest_feedback(self, trace_id: str, feedback: Any) -> None:
        """
        Enrich the graph with knowledge extracted from HITL feedback.

        Knowledge produced:
          - Incorrect diagnosis → MISDIAGNOSED_AS edge between services
          - New system relationships → new service edges
          - Runbook references → Runbook nodes linked to Incident
          - Reviewer modification → MODIFIED_BY edge on Incident node
        """
        incident_id = f"incident:{trace_id}"

        # Misdiagnosis knowledge
        if (
            feedback.diagnosis_accuracy == "incorrect"
            and feedback.correct_root_cause
        ):
            # The focal service was wrongly blamed; actual root cause is elsewhere
            focal_node = None
            for _, target, data in self._g.out_edges(incident_id, data=True):
                if data.get("rel") == "FOCAL_SERVICE":
                    focal_node = target
                    break

            if focal_node:
                correct_id = f"service:{feedback.correct_root_cause}"
                if not self._g.has_node(correct_id):
                    self._g.add_node(
                        correct_id,
                        node_type="service",
                        name=feedback.correct_root_cause,
                    )
                # Increment misdiagnosis count
                edge_data = self._g.get_edge_data(focal_node, correct_id) or {}
                count = edge_data.get("count", 0) + 1
                self._g.add_edge(
                    focal_node,
                    correct_id,
                    rel="MISDIAGNOSED_AS",
                    count=count,
                )
                logger.debug(
                    "[NetworkXGraphStore] Misdiagnosis edge: %s -> %s (count=%d)",
                    focal_node,
                    correct_id,
                    count,
                )

        # New system relationships from reviewer
        for relationship_str in feedback.new_system_relationships:
            # Expected format: "service_a -> service_b" or "service_a depends_on service_b"
            parts = [p.strip() for p in relationship_str.replace("->", "|").split("|")]
            if len(parts) == 2:
                src_id = f"service:{parts[0]}"
                tgt_id = f"service:{parts[1]}"
                for node_id, name in [(src_id, parts[0]), (tgt_id, parts[1])]:
                    if not self._g.has_node(node_id):
                        self._g.add_node(node_id, node_type="service", name=name)
                self._g.add_edge(src_id, tgt_id, rel="DEPENDS_ON", source="hitl_feedback")

        # Runbook references
        for ref in feedback.runbook_references:
            runbook_id = f"runbook:{ref}"
            if not self._g.has_node(runbook_id):
                self._g.add_node(runbook_id, node_type="runbook", url=ref)
            if self._g.has_node(incident_id):
                self._g.add_edge(incident_id, runbook_id, rel="REFERENCES")

        # Reviewer modification
        if feedback.decision == "modify" and feedback.reviewer_id:
            self._g.nodes[incident_id]["modified_by"] = feedback.reviewer_id
            self._g.nodes[incident_id]["modifications"] = str(
                feedback.modifications_applied
            )

        self._flush()
        logger.debug("[NetworkXGraphStore] Feedback ingested for trace %s", trace_id)

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_service_incident_history(
        self, service: str, limit: int = 10
    ) -> list[dict]:
        """Return past incidents involving a service (as focal or affected)."""
        service_id = f"service:{service}"
        if not self._g.has_node(service_id):
            return []

        history: list[dict] = []
        for incident_id, _, data in self._g.in_edges(service_id, data=True):
            node_data = self._g.nodes.get(incident_id, {})
            if node_data.get("node_type") != "incident":
                continue
            history.append({
                "trace_id": node_data.get("trace_id", ""),
                "incident_type": node_data.get("incident_type", ""),
                "outcome": node_data.get("outcome", ""),
                "diagnosis_confidence": node_data.get("diagnosis_confidence", 0.0),
                "symbolic_path": node_data.get("symbolic_path", ""),
                "relationship": data.get("rel", ""),
                "created_at": node_data.get("created_at", ""),
            })

        # Sort by creation time descending
        history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return history[:limit]

    def get_pattern_frequency(self, template_key: str) -> dict:
        """Get occurrence count and outcome distribution for a failure pattern."""
        pattern_id = f"pattern:{template_key}"
        if not self._g.has_node(pattern_id):
            return {"template_key": template_key, "count": 0, "outcomes": {}}

        count = self._g.nodes[pattern_id].get("count", 0)
        outcomes: dict[str, int] = {}

        for incident_id, _, data in self._g.in_edges(pattern_id, data=True):
            if data.get("rel") == "EXHIBITED":
                node = self._g.nodes.get(incident_id, {})
                outcome = node.get("outcome", "unknown")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

        return {
            "template_key": template_key,
            "count": count,
            "outcomes": outcomes,
        }

    def get_causal_chains(
        self, service: str, depth: int = 2
    ) -> list[list[str]]:
        """
        Retrieve known causal chains where this service is the root cause.

        Returns paths: [root_cause_service, ..., affected_service]
        """
        service_id = f"service:{service}"
        if not self._g.has_node(service_id):
            return []

        chains: list[list[str]] = []
        # Find incidents where this service is root cause
        for incident_id, _, data in self._g.in_edges(service_id, data=True):
            if data.get("rel") != "ROOT_CAUSE":
                continue
            # Collect affected services from this incident
            for _, affected_id, edge_data in self._g.out_edges(incident_id, data=True):
                if edge_data.get("rel") == "AFFECTED":
                    affected_node = self._g.nodes.get(affected_id, {})
                    affected_name = affected_node.get("name", affected_id)
                    chains.append([service, affected_name])

        return chains[:10]  # Cap to avoid context bloat

    def get_misdiagnosis_history(self, service: str) -> list[dict]:
        """Get past misdiagnoses where this service was wrongly identified as root cause."""
        service_id = f"service:{service}"
        if not self._g.has_node(service_id):
            return []

        misdiagnoses: list[dict] = []
        for _, target_id, data in self._g.out_edges(service_id, data=True):
            if data.get("rel") == "MISDIAGNOSED_AS":
                target_node = self._g.nodes.get(target_id, {})
                misdiagnoses.append({
                    "wrongly_blamed": service,
                    "actual_root_cause": target_node.get("name", target_id),
                    "count": data.get("count", 1),
                })

        return sorted(misdiagnoses, key=lambda x: x["count"], reverse=True)

    # ── Graph statistics ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return basic graph statistics for diagnostics."""
        node_types: dict[str, int] = {}
        for _, data in self._g.nodes(data=True):
            t = data.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            "node_types": node_types,
            "persist_path": str(self._path),
        }

    def close(self) -> None:
        """Flush graph to disk on shutdown."""
        self._flush()
        logger.info("[NetworkXGraphStore] Closed and flushed.")
