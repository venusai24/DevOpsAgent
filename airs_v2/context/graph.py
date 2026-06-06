"""
airs_v2/context/graph.py
=========================

ContextGraph — Infrastructure Dependency Graph
-----------------------------------------------

The central data structure of Stage 2. Wraps a ``networkx.DiGraph`` with
typed Pydantic v2 node/edge attributes and exposes a rich query API that
the MCP server turns into tools.

Topology model
--------------
Nodes represent infrastructure entities: Services, Databases, Caches,
External dependencies. Edges are directed DEPENDS_ON relationships
(``A → B`` means "A depends on B").

Blast radius semantics
-----------------------
For a focal ``service`` the blast radius at ``depth=2`` is the union of:

* **Downstream** (dependency cone) — nodes that *service* transitively
  calls, up to 2 hops. A failure here propagates *up* to the focal service.
* **Upstream** (impact cone) — nodes that transitively call *service*, up
  to 2 hops. A failure in the focal service propagates *up* to these callers.

Example: focal = ``payments-db`` (depth=2):
  Downstream: ∅ (payments-db has no outgoing DEPENDS_ON edges)
  Upstream:   payments-service, payment-processor (depth-1)
              order-processing-api, api-gateway (depth-2 via payments-service)

This is the set returned by ``subgraph()`` and all MCP tools — no other
nodes ever appear in any response.

Seeding
-------
The graph is seeded from ``mock_enterprise/topology_fixtures.json``, the
same fixture used by the v1 EKG. Zero data duplication.

Thread-safety
-------------
``ContextGraph`` is not thread-safe for concurrent ``update_health`` calls.
In the async LangGraph context, all mutations happen inside a single
coroutine (the topology_agent_node), so no locking is required in Stage 2.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default fixtures path — relative to this file's package root
# ---------------------------------------------------------------------------
_FIXTURES_PATH = (
    Path(__file__).resolve().parents[2]   # DevOpsAgent/
    / "mock_enterprise"
    / "topology_fixtures.json"
)

# Blast radius depth — hardcoded per Stage 2 design decision
BLAST_RADIUS_DEPTH = 2


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class NodeSnapshot(BaseModel):
    """Point-in-time snapshot of a single infrastructure node."""
    name: str
    node_type: str                          # service | database | cache | external
    tier: int
    health_status: str                      # healthy | degraded | critical | unknown
    owner: str
    namespace: str
    on_call: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    captured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MetricsSnapshot(BaseModel):
    """
    Live-ish metrics for a single infrastructure node.

    ``stale=True`` signals to the reasoning engine that these values come
    from the static topology fixture and have not been refreshed from the
    live mock API. Stage 3 will flip this to False when real data is fetched.
    The reasoning engine must apply lower confidence weights when stale=True.
    """
    service: str
    connection_saturation: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="current_connections / max_connections (0.0–1.0)",
    )
    error_rate_pct: float = Field(
        default=0.0,
        ge=0.0, le=100.0,
        description="Current HTTP/query error rate %",
    )
    replica_desired: int = Field(default=1, description="Desired pod replica count")
    replica_ready: int = Field(default=1, description="Currently ready replicas")
    memory_usage_pct: float = Field(
        default=0.0,
        ge=0.0, le=100.0,
        description="Memory usage % (caches only)",
    )
    stale: bool = Field(
        default=True,
        description=(
            "True = values from static fixture; False = refreshed from live API. "
            "Reasoning engine should weight stale metrics with lower confidence."
        ),
    )
    captured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CorrelationRecord(BaseModel):
    """A known historical co-failure relationship."""
    partner_service: str
    description: str
    historical_incident_ids: list[str] = Field(default_factory=list)


class EdgeSnapshot(BaseModel):
    """A directed dependency edge between two nodes."""
    source: str
    target: str
    protocol: str = "http"
    error_rate_pct: float = 0.0
    latency_p99_ms: float = 0.0
    is_critical_path: bool = True


class ContextSubgraph(BaseModel):
    """
    A bounded subgraph representing the topological blast radius of an incident.

    ``nodes`` and ``edges`` are limited to ``BLAST_RADIUS_DEPTH`` hops from
    ``focus_service``.  The reasoning engine must not assume any node absent
    from this subgraph is unaffected — it simply falls outside the bounded
    context window.
    """
    focus_service: str
    depth: int = BLAST_RADIUS_DEPTH
    nodes: list[NodeSnapshot] = Field(default_factory=list)
    edges: list[EdgeSnapshot] = Field(default_factory=list)
    affected_count: int = 0
    tier1_node_names: list[str] = Field(default_factory=list)
    captured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_markdown(self) -> str:
        lines = [
            f"## Context Subgraph: `{self.focus_service}` (depth={self.depth})",
            f"**Bounded context**: {self.affected_count} nodes, "
            f"{len(self.edges)} edges",
        ]
        if self.tier1_node_names:
            lines.append(
                f"⚠️  **Tier-1 nodes in blast radius**: "
                f"{', '.join(f'`{n}`' for n in self.tier1_node_names)}"
            )
        lines.append("\n### Nodes")
        icon_map = {
            "healthy": "🟢", "degraded": "🟡",
            "critical": "🔴", "unknown": "⚪",
        }
        for node in self.nodes:
            icon = icon_map.get(node.health_status, "⚪")
            tier1 = " ⚠️T1" if node.tier == 1 else ""
            lines.append(
                f"- {icon} **{node.name}** "
                f"[{node.node_type}][Tier {node.tier}]{tier1} "
                f"— {node.health_status.upper()}"
            )
        lines.append("\n### Dependencies")
        for edge in self.edges:
            warn = " ⚠️" if edge.error_rate_pct > 5 else ""
            lines.append(
                f"- `{edge.source}` → `{edge.target}` "
                f"({edge.protocol}, err={edge.error_rate_pct:.1f}%){warn}"
            )
        return "\n".join(lines)


class BlastRadiusResult(BaseModel):
    """Structured blast radius analysis for a focal service."""
    focus_service: str
    depth: int = BLAST_RADIUS_DEPTH
    affected_services: list[str] = Field(default_factory=list)
    tier1_services: list[str] = Field(default_factory=list)
    tier1_impact: bool = False
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    propagation_paths: list[list[str]] = Field(default_factory=list)
    on_call_contacts: list[str] = Field(default_factory=list)
    captured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# ContextGraph
# ---------------------------------------------------------------------------

class ContextGraph:
    """
    Infrastructure dependency graph for AIRS v2.

    This is the single source of truth for infrastructure topology during
    an incident response session.  The MCP server holds one singleton
    instance; the reasoning engine never queries topology directly.

    Parameters
    ----------
    fixtures_path:
        Path to ``topology_fixtures.json``. Defaults to the mock_enterprise
        fixture bundled with the repository.
    """

    def __init__(self, fixtures_path: Optional[Path] = None) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._correlations: list[dict] = []
        self._path = fixtures_path or _FIXTURES_PATH
        self._loaded = False
        self.load_from_fixtures(self._path)

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    def load_from_fixtures(self, path: Path) -> None:
        """Seed the graph from a topology JSON fixture file."""
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("[ContextGraph] Could not load fixtures from %s: %s", path, exc)
            return

        services = data.get("services", [])
        for raw in services:
            name = raw["name"]
            attrs: dict[str, Any] = {
                "node_type": raw.get("type", "service"),
                "tier": raw.get("tier", 3),
                "health_status": raw.get("health_status", "unknown"),
                "owner": raw.get("owner", "unknown"),
                "namespace": raw.get("namespace", "prod"),
                "on_call": raw.get("on_call", ""),
                # Type-specific extras stored flat in metadata
                "engine": raw.get("engine", ""),
                "max_connections": raw.get("max_connections", 0),
                "current_connections": raw.get("current_connections", 0),
                "replicas": raw.get("replicas", 1),
                "memory_usage_pct": raw.get("memory_usage_pct", 0.0),
                "error_rate_pct": raw.get("error_rate_pct", 0.0),
            }
            self._g.add_node(name, **attrs)

        # Add directed edges: source → target means "source DEPENDS_ON target"
        for raw in services:
            source = raw["name"]
            for dep in raw.get("depends_on", []):
                if not self._g.has_node(dep):
                    # Add stub node for undeclared dependencies
                    self._g.add_node(dep, node_type="service", tier=3,
                                     health_status="unknown", owner="unknown",
                                     namespace="prod", on_call="", engine="",
                                     max_connections=0, current_connections=0,
                                     replicas=1, memory_usage_pct=0.0,
                                     error_rate_pct=0.0)
                self._g.add_edge(source, dep,
                                 protocol="http",
                                 error_rate_pct=0.0,
                                 latency_p99_ms=0.0,
                                 is_critical_path=True)

        self._correlations = data.get("known_failure_correlations", [])
        self._loaded = True
        logger.info(
            "[ContextGraph] Loaded %d nodes, %d edges from %s",
            self._g.number_of_nodes(), self._g.number_of_edges(), path.name,
        )

    # ------------------------------------------------------------------
    # Core topology queries
    # ------------------------------------------------------------------

    def node_names(self) -> list[str]:
        """All node names currently in the graph."""
        return list(self._g.nodes())

    def get_node(self, service: str) -> Optional[NodeSnapshot]:
        """Return a point-in-time snapshot of *service*, or None if unknown."""
        if not self._g.has_node(service):
            return None
        attrs = self._g.nodes[service]
        return NodeSnapshot(
            name=service,
            node_type=attrs.get("node_type", "service"),
            tier=attrs.get("tier", 3),
            health_status=attrs.get("health_status", "unknown"),
            owner=attrs.get("owner", "unknown"),
            namespace=attrs.get("namespace", "prod"),
            on_call=attrs.get("on_call", ""),
            metadata={
                k: v for k, v in attrs.items()
                if k not in ("node_type", "tier", "health_status",
                             "owner", "namespace", "on_call")
            },
        )

    def update_health(self, service: str, status: str) -> bool:
        """
        Mutate the health_status of *service* in-place.

        Returns True on success, False if the service is not in the graph.
        The MCP server calls this when the perception layer detects a
        health change so downstream queries reflect current state.
        """
        valid = {"healthy", "degraded", "critical", "unknown"}
        if status not in valid:
            raise ValueError(f"Invalid health status '{status}'. Must be one of {valid}")
        if not self._g.has_node(service):
            logger.warning("[ContextGraph] update_health: node '%s' not found", service)
            return False
        self._g.nodes[service]["health_status"] = status
        logger.debug("[ContextGraph] %s health → %s", service, status)
        return True

    def get_metrics_snapshot(self, service: str) -> Optional[MetricsSnapshot]:
        """
        Return live-ish metrics for *service*.

        ``stale=True`` until Stage 3 connects the live mock API.
        """
        if not self._g.has_node(service):
            return None
        attrs = self._g.nodes[service]
        max_conn = attrs.get("max_connections", 0)
        curr_conn = attrs.get("current_connections", 0)
        saturation = (curr_conn / max_conn) if max_conn > 0 else 0.0
        return MetricsSnapshot(
            service=service,
            connection_saturation=round(min(saturation, 1.0), 4),
            error_rate_pct=attrs.get("error_rate_pct", 0.0),
            replica_desired=attrs.get("replicas", 1),
            replica_ready=attrs.get("replicas", 1),   # static fixture — same value
            memory_usage_pct=attrs.get("memory_usage_pct", 0.0),
            stale=True,   # Stage 2: always stale until live API connected
        )

    def get_failure_correlations(self, service: str) -> list[CorrelationRecord]:
        """
        Return historical co-failure records involving *service*.

        Scans ``known_failure_correlations`` from the topology fixture and
        returns every record where *service* appears in the services list.
        """
        records: list[CorrelationRecord] = []
        for entry in self._correlations:
            partners = entry.get("services", [])
            if service not in partners:
                continue
            others = [s for s in partners if s != service]
            for partner in others:
                records.append(CorrelationRecord(
                    partner_service=partner,
                    description=entry.get("description", ""),
                    historical_incident_ids=entry.get("historical_incident_ids", []),
                ))
        return records

    # ------------------------------------------------------------------
    # Blast radius traversal (depth = BLAST_RADIUS_DEPTH = 2)
    # ------------------------------------------------------------------

    def _bfs_bounded(
        self,
        start: str,
        follow_edges: str,      # "successors" | "predecessors"
        depth: int,
    ) -> set[str]:
        """
        Bounded BFS returning all nodes reachable within *depth* hops from *start*.

        ``follow_edges="successors"``  — follows outgoing edges (DEPENDS_ON cone,
                                          i.e., things the focal service calls).
        ``follow_edges="predecessors"``— follows incoming edges (impact cone,
                                          i.e., things that call the focal service).
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        get_neighbors = (
            self._g.successors if follow_edges == "successors"
            else self._g.predecessors
        )
        while queue:
            node, d = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if d < depth:
                for neighbor in get_neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))
        visited.discard(start)   # The focal service itself is excluded from the set
        return visited

    def blast_radius(self, service: str) -> BlastRadiusResult:
        """
        Compute the blast radius of *service* at the fixed depth of 2.

        The blast radius is the union of:
        - Downstream cone: nodes this service depends on (may cascade up)
        - Upstream cone: nodes that depend on this service (directly impacted)
        """
        if not self._g.has_node(service):
            return BlastRadiusResult(focus_service=service)

        downstream = self._bfs_bounded(service, "successors", BLAST_RADIUS_DEPTH)
        upstream = self._bfs_bounded(service, "predecessors", BLAST_RADIUS_DEPTH)
        affected = downstream | upstream

        tier1 = [
            n for n in affected
            if self._g.nodes[n].get("tier", 3) == 1
        ]

        # Propagation paths: all simple paths from focal to tier-1 callers
        paths: list[list[str]] = []
        for t1 in tier1:
            if self._g.has_node(t1):
                try:
                    for path in nx.all_simple_paths(
                        self._g.reverse(), service, t1, cutoff=BLAST_RADIUS_DEPTH
                    ):
                        paths.append(list(reversed(path)))
                except nx.NetworkXError:
                    pass

        # Collect unique on-call contacts
        contacts: set[str] = set()
        for n in affected:
            oc = self._g.nodes[n].get("on_call", "")
            if oc and oc != "N/A":
                contacts.add(oc)

        # Risk score: weighted by tier-1 presence + proportion of affected tier-1
        base = len(tier1) / max(1, len(affected))
        risk = min(1.0, round(0.4 + 0.6 * base if tier1 else base * 0.3, 3))

        return BlastRadiusResult(
            focus_service=service,
            depth=BLAST_RADIUS_DEPTH,
            affected_services=sorted(affected),
            tier1_services=sorted(tier1),
            tier1_impact=bool(tier1),
            risk_score=risk,
            propagation_paths=paths,
            on_call_contacts=sorted(contacts),
        )

    def subgraph(self, service: str) -> ContextSubgraph:
        """
        Return the node + edge set bounded to depth-2 blast radius of *service*.

        This is the *only* topology data the reasoning engine is allowed to
        consume. Nodes outside this subgraph are outside the context window.
        """
        if not self._g.has_node(service):
            return ContextSubgraph(focus_service=service)

        br = self.blast_radius(service)
        in_scope: set[str] = set(br.affected_services) | {service}

        nodes: list[NodeSnapshot] = []
        for name in in_scope:
            snap = self.get_node(name)
            if snap:
                nodes.append(snap)

        edges: list[EdgeSnapshot] = []
        for src, dst, edata in self._g.edges(data=True):
            if src in in_scope and dst in in_scope:
                edges.append(EdgeSnapshot(
                    source=src,
                    target=dst,
                    protocol=edata.get("protocol", "http"),
                    error_rate_pct=edata.get("error_rate_pct", 0.0),
                    latency_p99_ms=edata.get("latency_p99_ms", 0.0),
                    is_critical_path=edata.get("is_critical_path", True),
                ))

        return ContextSubgraph(
            focus_service=service,
            depth=BLAST_RADIUS_DEPTH,
            nodes=nodes,
            edges=edges,
            affected_count=len(in_scope),
            tier1_node_names=br.tier1_services,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "node_count": self._g.number_of_nodes(),
            "edge_count": self._g.number_of_edges(),
            "loaded": self._loaded,
            "correlation_records": len(self._correlations),
            "blast_radius_depth": BLAST_RADIUS_DEPTH,
        }
