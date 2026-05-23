"""
agent/reasoning/knowledge_graph.py

Enterprise Knowledge Graph (EKG) client using Neo4j as the production backend
and NetworkX as an in-memory fallback for demo/testing.

The EKG models the infrastructure as a directed graph where:
  Nodes  — Services, Databases, Caches, External dependencies
  Edges  — DEPENDS_ON relationships with live health metadata

Core capabilities:
  - Load topology from fixtures or Neo4j at startup
  - Traverse upstream/downstream dependencies to any depth
  - Calculate blast radius: all paths from a service to tier-1 critical assets
  - Update node health status in real-time as incidents are detected
  - Find correlated failures from topology_fixtures.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel

from agent.reasoning.graph_schema import (
    BaseNode,
    BlastRadiusReport,
    CacheNode,
    DatabaseNode,
    DependencyEdge,
    ExternalNode,
    HealthStatus,
    NodeType,
    ServiceNode,
    SubGraph,
)
from config import settings

logger = logging.getLogger(__name__)

# Optional Neo4j driver — only imported if NEO4J_URI is configured
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    HAS_NEO4J = True
except ImportError:
    AsyncGraphDatabase = None  # type: ignore
    AsyncDriver = None  # type: ignore
    HAS_NEO4J = False


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def _build_node(raw: dict[str, Any]) -> BaseNode:
    """
    Construct the correct BaseNode subclass from a raw fixture dict.
    Discriminates on the 'type' field.
    """
    node_type = raw.get("type", "service")
    common = {
        "name": raw["name"],
        "tier": raw.get("tier", 3),
        "health_status": HealthStatus(raw.get("health_status", "unknown")),
        "owner": raw.get("owner", "unknown"),
        "namespace": raw.get("namespace", "prod"),
        "metadata": {k: v for k, v in raw.items() if k not in (
            "name", "type", "tier", "health_status", "owner", "namespace", "depends_on"
        )},
    }
    if node_type == "database":
        return DatabaseNode(
            **common,
            engine=raw.get("engine", "postgresql"),
            max_connections=raw.get("max_connections", 100),
            current_connections=raw.get("current_connections", 0),
        )
    if node_type == "cache":
        return CacheNode(**common, engine=raw.get("engine", "redis"))
    if node_type == "external":
        return ExternalNode(**common, endpoint=raw.get("endpoint", ""))
    return ServiceNode(
        **common,
        replicas=raw.get("replicas", 1),
        on_call=raw.get("on_call", ""),
    )


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """
    Enterprise Knowledge Graph providing deterministic, graph-structured
    access to infrastructure topology.

    Backends:
      - Neo4j (production): Persistent, supports Cypher queries.
      - NetworkX DiGraph (demo): In-memory, no external service required.

    Usage:
        ekg = KnowledgeGraph.get_instance()
        await ekg.initialize()
        subgraph = await ekg.get_dependency_chain("payments-service", depth=2)
    """

    _instance: Optional["KnowledgeGraph"] = None

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._neo4j_driver: Optional[Any] = None
        self._initialized: bool = False
        self._topology_path = Path(settings.TOPOLOGY_FIXTURES_PATH)
        self._on_call_map: dict[str, str] = {}       # service_name -> on_call_email
        self._failure_correlations: list[dict] = []

    @classmethod
    def get_instance(cls) -> "KnowledgeGraph":
        """Return the singleton KnowledgeGraph instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load topology from fixtures and optionally sync to Neo4j."""
        if self._initialized:
            return

        if not self._topology_path.exists():
            logger.warning("[EKG] Topology fixtures not found at %s", self._topology_path)
            self._initialized = True
            return

        with self._topology_path.open() as f:
            topology = json.load(f)

        await self.load_topology(topology)

        if settings.NEO4J_URI and HAS_NEO4J:
            await self._init_neo4j(topology)

        self._initialized = True
        logger.info(
            "[EKG] Initialized: %d nodes, %d edges.",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    async def load_topology(self, topology: dict[str, Any]) -> None:
        """
        Hydrate the in-memory NetworkX graph from a topology dict.

        The topology dict must follow the schema in topology_fixtures.json:
          - 'services': list of node definitions with optional 'depends_on' lists
          - 'on_call_schedules': list of on-call rotation entries
          - 'known_failure_correlations': list of correlated-failure patterns
        """
        # Register nodes
        for raw in topology.get("services", []):
            node = _build_node(raw)
            self._graph.add_node(
                node.name,
                **node.model_dump(mode="json"),
            )
            # Build on-call index
            if hasattr(node, "on_call") and node.on_call:
                self._on_call_map[node.name] = node.on_call

        # Register dependency edges
        for raw in topology.get("services", []):
            source = raw["name"]
            for dep in raw.get("depends_on", []):
                if self._graph.has_node(dep):
                    self._graph.add_edge(
                        source, dep,
                        edge_type="DEPENDS_ON",
                        error_rate_pct=0.0,
                        latency_p99_ms=0.0,
                    )

        # Cache on-call schedules
        for schedule in topology.get("on_call_schedules", []):
            pattern = schedule.get("service_pattern", "").rstrip("*")
            for node_name in self._graph.nodes:
                if node_name.startswith(pattern):
                    self._on_call_map[node_name] = schedule.get("engineer", "")

        # Cache failure correlations for quick lookup
        self._failure_correlations = topology.get("known_failure_correlations", [])

    async def _init_neo4j(self, topology: dict[str, Any]) -> None:
        """Sync the topology into Neo4j using Cypher MERGE statements."""
        try:
            self._neo4j_driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            async with self._neo4j_driver.session() as session:
                # Create/merge all nodes
                for raw in topology.get("services", []):
                    await session.run(
                        "MERGE (n:Service {name: $name}) "
                        "SET n.tier = $tier, n.health_status = $health, n.type = $type",
                        name=raw["name"],
                        tier=raw.get("tier", 3),
                        health=raw.get("health_status", "unknown"),
                        type=raw.get("type", "service"),
                    )
                # Create DEPENDS_ON relationships
                for raw in topology.get("services", []):
                    for dep in raw.get("depends_on", []):
                        await session.run(
                            "MATCH (a:Service {name: $src}), (b:Service {name: $tgt}) "
                            "MERGE (a)-[:DEPENDS_ON]->(b)",
                            src=raw["name"], tgt=dep,
                        )
            logger.info("[EKG] Neo4j synchronized with %d services.", len(topology.get("services", [])))
        except Exception as exc:
            logger.warning("[EKG] Neo4j sync failed (%s) — graph running from NetworkX only.", exc)

    # ------------------------------------------------------------------
    # Graph Queries
    # ------------------------------------------------------------------

    async def get_node(self, service: str) -> Optional[dict[str, Any]]:
        """Return the node attributes dict for a given service name."""
        await self.initialize()
        if self._graph.has_node(service):
            return dict(self._graph.nodes[service])
        return None

    async def get_dependencies(self, service: str) -> list[dict[str, Any]]:
        """
        Return the direct downstream dependencies of a service.
        (i.e., the services that ``service`` DEPENDS_ON).
        """
        await self.initialize()
        if not self._graph.has_node(service):
            return []
        return [
            {**dict(self._graph.nodes[nb]), "name": nb}
            for nb in self._graph.successors(service)
        ]

    async def get_dependents(self, service: str) -> list[dict[str, Any]]:
        """
        Return all services that directly depend ON ``service``.
        (Reverse traversal — used for blast radius computation.)
        """
        await self.initialize()
        if not self._graph.has_node(service):
            return []
        return [
            {**dict(self._graph.nodes[nb]), "name": nb}
            for nb in self._graph.predecessors(service)
        ]

    async def get_dependency_chain(self, service: str, depth: int = 2) -> SubGraph:
        """
        Extract a bounded subgraph containing all nodes reachable from
        ``service`` within ``depth`` hops (downstream dependencies).

        Returns a SubGraph instance with serialized nodes and edges.
        """
        await self.initialize()
        if not self._graph.has_node(service):
            return SubGraph(focus_service=service, nodes=[], edges=[], depth=depth)

        # BFS to collect nodes within depth
        visited: set[str] = {service}
        frontier: set[str] = {service}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for successor in self._graph.successors(node):
                    if successor not in visited:
                        visited.add(successor)
                        next_frontier.add(successor)
            frontier = next_frontier

        subgraph = self._graph.subgraph(visited)
        nodes = [{**dict(subgraph.nodes[n]), "name": n} for n in subgraph.nodes]
        edges = [
            {
                "source": u,
                "target": v,
                **dict(subgraph.edges[u, v]),
            }
            for u, v in subgraph.edges
        ]
        return SubGraph(focus_service=service, nodes=nodes, edges=edges, depth=depth)

    async def calculate_blast_radius(self, service: str) -> BlastRadiusReport:
        """
        Compute the blast radius of a change or failure at ``service``.

        Traverses the graph in reverse (dependents) to find all services
        that would be impacted. Scores the risk based on tier-1 presence
        and number of propagation paths.
        """
        await self.initialize()

        # Reverse BFS from the failing service
        affected: set[str] = set()
        queue = [service]
        while queue:
            current = queue.pop(0)
            for dependent in self._graph.predecessors(current):
                if dependent not in affected:
                    affected.add(dependent)
                    queue.append(dependent)

        # Classify by tier
        tier1_services = [
            svc for svc in affected
            if self._graph.has_node(svc) and self._graph.nodes[svc].get("tier", 3) == 1
        ]

        # Find all simple paths to tier-1 services using the reverse graph
        rev_graph = self._graph.reverse()
        propagation_paths: list[list[str]] = []
        for t1 in tier1_services:
            try:
                for path in nx.all_simple_paths(rev_graph, source=service, target=t1, cutoff=4):
                    propagation_paths.append(path)
                    if len(propagation_paths) >= 10:  # cap for performance
                        break
            except nx.NetworkXNoPath:
                pass

        # Risk scoring
        risk_score = min(1.0, (
            (0.4 if tier1_services else 0.0) +
            (0.3 * min(1.0, len(affected) / 10)) +
            (0.3 * min(1.0, len(propagation_paths) / 5))
        ))

        if risk_score >= 0.7:
            recommendation = "block"
        elif risk_score >= 0.3 or tier1_services:
            recommendation = "require_approval"
        else:
            recommendation = "auto_execute"

        # Estimate user impact based on tier-1 presence and affected service count
        user_impact = min(100.0, len(tier1_services) * 30.0 + len(affected) * 5.0)

        # Gather on-call contacts for affected services
        contacts = list({
            self._on_call_map[svc]
            for svc in affected
            if svc in self._on_call_map
        })

        return BlastRadiusReport(
            target_service=service,
            affected_services=sorted(affected),
            tier1_services=sorted(tier1_services),
            tier1_impact=bool(tier1_services),
            propagation_paths=propagation_paths,
            risk_score=round(risk_score, 2),
            recommendation=recommendation,
            on_call_contacts=contacts,
            estimated_user_impact_pct=round(user_impact, 1),
        )

    async def update_node_health(self, service: str, status: HealthStatus) -> None:
        """Update the live health status of a node in the in-memory graph."""
        await self.initialize()
        if self._graph.has_node(service):
            self._graph.nodes[service]["health_status"] = status.value
            logger.debug("[EKG] Updated %s health → %s", service, status.value)

    async def get_failure_correlations(self, service: str) -> list[dict[str, Any]]:
        """
        Return known failure correlation groups that include ``service``.
        These are pre-defined patterns from topology_fixtures.json.
        """
        await self.initialize()
        return [
            corr for corr in self._failure_correlations
            if service in corr.get("services", [])
        ]

    async def get_on_call(self, service: str) -> Optional[str]:
        """Return the on-call engineer for the given service."""
        await self.initialize()
        return self._on_call_map.get(service)

    def all_service_names(self) -> list[str]:
        """Return all node names currently in the graph."""
        return list(self._graph.nodes)
