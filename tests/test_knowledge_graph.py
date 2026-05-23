"""
tests/test_knowledge_graph.py

Unit tests for the Enterprise Knowledge Graph (EKG) client.

Tests cover:
  - Singleton pattern
  - In-memory NetworkX graph initialization from topology fixtures
  - Dependency chain traversal (BFS → SubGraph)
  - Blast radius calculation (reverse BFS → BlastRadiusReport)
  - Health status update
  - get_node / get_dependencies / get_dependents
  - on-call map construction
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from agent.reasoning.graph_schema import HealthStatus, BlastRadiusReport, SubGraph
from agent.reasoning.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Minimal topology fixture
# ---------------------------------------------------------------------------

MINIMAL_TOPOLOGY = {
    "services": [
        {
            "name": "checkout-service",
            "type": "service",
            "tier": 1,
            "health_status": "healthy",
            "owner": "payments-team",
            "namespace": "prod",
            "on_call": "alice@example.com",
            "depends_on": ["payments-service"],
        },
        {
            "name": "payments-service",
            "type": "service",
            "tier": 1,
            "health_status": "healthy",
            "owner": "payments-team",
            "namespace": "prod",
            "on_call": "alice@example.com",
            "depends_on": ["payments-db", "redis-cache"],
        },
        {
            "name": "payments-db",
            "type": "database",
            "tier": 1,
            "health_status": "healthy",
            "owner": "dba-team",
            "namespace": "prod",
            "engine": "postgresql",
            "max_connections": 100,
            "depends_on": [],
        },
        {
            "name": "redis-cache",
            "type": "cache",
            "tier": 2,
            "health_status": "healthy",
            "owner": "platform-team",
            "namespace": "prod",
            "depends_on": [],
        },
        {
            "name": "analytics-service",
            "type": "service",
            "tier": 3,
            "health_status": "healthy",
            "owner": "data-team",
            "namespace": "prod",
            "depends_on": ["payments-db"],
        },
    ],
    "on_call_schedules": [],
    "known_failure_correlations": [
        {
            "pattern": "db_cascade",
            "services": ["payments-service", "checkout-service", "payments-db"],
            "description": "DB failure cascades to all upstream services",
        }
    ],
    "historical_incidents": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ekg_singleton():
    """Reset KnowledgeGraph singleton before each test for isolation."""
    KnowledgeGraph._instance = None
    yield
    if KnowledgeGraph._instance is not None:
        # Close any Neo4j driver if open
        instance = KnowledgeGraph._instance
        if instance._neo4j_driver is not None:
            pass  # Would call close() in production
    KnowledgeGraph._instance = None


@pytest.fixture
async def ekg():
    """KnowledgeGraph pre-loaded with MINIMAL_TOPOLOGY (no file I/O needed)."""
    graph = KnowledgeGraph.get_instance()
    await graph.load_topology(MINIMAL_TOPOLOGY)
    graph._initialized = True
    return graph


# ---------------------------------------------------------------------------
# Tests: Singleton pattern
# ---------------------------------------------------------------------------

def test_singleton_pattern():
    """get_instance() always returns the same object."""
    g1 = KnowledgeGraph.get_instance()
    g2 = KnowledgeGraph.get_instance()
    assert g1 is g2


# ---------------------------------------------------------------------------
# Tests: initialize from file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialize_from_fixture_file(tmp_path, monkeypatch):
    """initialize() loads topology from a JSON file without error."""
    fixture_file = tmp_path / "topology_fixtures.json"
    fixture_file.write_text(json.dumps(MINIMAL_TOPOLOGY))

    graph = KnowledgeGraph.get_instance()
    monkeypatch.setattr(
        "agent.reasoning.knowledge_graph.settings.TOPOLOGY_FIXTURES_PATH",
        str(fixture_file),
    )
    monkeypatch.setattr(
        "agent.reasoning.knowledge_graph.settings.NEO4J_URI",
        None,
    )
    # Patch _topology_path to point to our temp file
    graph._topology_path = fixture_file

    await graph.initialize()

    assert graph._initialized
    assert graph._graph.number_of_nodes() == len(MINIMAL_TOPOLOGY["services"])


@pytest.mark.asyncio
async def test_initialize_missing_file():
    """initialize() with a missing fixture file does not raise."""
    graph = KnowledgeGraph.get_instance()
    from pathlib import Path
    graph._topology_path = Path("/nonexistent/topology.json")
    await graph.initialize()  # Should complete without error
    assert graph._initialized


# ---------------------------------------------------------------------------
# Tests: load_topology
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_topology_node_count(ekg):
    """load_topology creates all 5 expected nodes."""
    assert ekg._graph.number_of_nodes() == 5


@pytest.mark.asyncio
async def test_load_topology_edge_count(ekg):
    """load_topology creates the correct number of dependency edges."""
    # checkout→payments, payments→db, payments→redis, analytics→db = 4 edges
    assert ekg._graph.number_of_edges() == 4


@pytest.mark.asyncio
async def test_load_topology_node_attributes(ekg):
    """Node attributes are correctly populated from fixture."""
    attrs = ekg._graph.nodes["payments-db"]
    assert attrs["tier"] == 1
    assert attrs["owner"] == "dba-team"


@pytest.mark.asyncio
async def test_load_topology_failure_correlations(ekg):
    """Known failure correlations are cached on the graph instance."""
    corrs = ekg._failure_correlations
    assert len(corrs) == 1
    assert "payments-db" in corrs[0]["services"]


# ---------------------------------------------------------------------------
# Tests: get_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_node_existing(ekg):
    """get_node returns the attribute dict for a known node."""
    node = await ekg.get_node("payments-db")
    assert node is not None
    assert node["tier"] == 1


@pytest.mark.asyncio
async def test_get_node_missing(ekg):
    """get_node returns None for an unknown service."""
    node = await ekg.get_node("nonexistent-svc")
    assert node is None


# ---------------------------------------------------------------------------
# Tests: get_dependencies / get_dependents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_dependencies_payments(ekg):
    """payments-service depends on payments-db and redis-cache."""
    deps = await ekg.get_dependencies("payments-service")
    dep_names = {d["name"] for d in deps}
    assert "payments-db" in dep_names
    assert "redis-cache" in dep_names


@pytest.mark.asyncio
async def test_get_dependencies_leaf(ekg):
    """payments-db has no outgoing dependencies."""
    deps = await ekg.get_dependencies("payments-db")
    assert deps == []


@pytest.mark.asyncio
async def test_get_dependents_payments_db(ekg):
    """payments-db is depended on by payments-service and analytics-service."""
    dependents = await ekg.get_dependents("payments-db")
    dep_names = {d["name"] for d in dependents}
    assert "payments-service" in dep_names
    assert "analytics-service" in dep_names


@pytest.mark.asyncio
async def test_get_dependents_leaf(ekg):
    """checkout-service has no upstream services depending on it."""
    dependents = await ekg.get_dependents("checkout-service")
    assert dependents == []


# ---------------------------------------------------------------------------
# Tests: get_dependency_chain (BFS traversal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_chain_depth1(ekg):
    """Depth-1 chain from payments-service includes direct deps only."""
    subgraph = await ekg.get_dependency_chain("payments-service", depth=1)
    assert isinstance(subgraph, SubGraph)
    node_names = {n["name"] for n in subgraph.nodes}
    assert "payments-service" in node_names
    assert "payments-db" in node_names
    assert "redis-cache" in node_names
    # checkout-service is upstream — should NOT appear in downstream deps
    assert "checkout-service" not in node_names


@pytest.mark.asyncio
async def test_dependency_chain_depth2(ekg):
    """Depth-2 chain from checkout-service reaches db and redis."""
    subgraph = await ekg.get_dependency_chain("checkout-service", depth=2)
    node_names = {n["name"] for n in subgraph.nodes}
    assert "checkout-service" in node_names
    assert "payments-service" in node_names
    assert "payments-db" in node_names
    assert "redis-cache" in node_names


@pytest.mark.asyncio
async def test_dependency_chain_unknown_service(ekg):
    """Unknown service returns a SubGraph with 0 nodes (no error)."""
    subgraph = await ekg.get_dependency_chain("nonexistent-svc", depth=2)
    assert isinstance(subgraph, SubGraph)
    assert subgraph.focus_service == "nonexistent-svc"
    assert len(subgraph.nodes) == 0


@pytest.mark.asyncio
async def test_dependency_chain_focus_service(ekg):
    """SubGraph.focus_service matches the requested service."""
    subgraph = await ekg.get_dependency_chain("payments-service", depth=1)
    assert subgraph.focus_service == "payments-service"


@pytest.mark.asyncio
async def test_dependency_chain_has_edges(ekg):
    """SubGraph.edges is a list (may be empty for isolated nodes)."""
    subgraph = await ekg.get_dependency_chain("payments-service", depth=1)
    assert isinstance(subgraph.edges, list)


# ---------------------------------------------------------------------------
# Tests: calculate_blast_radius
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blast_radius_returns_report(ekg):
    """calculate_blast_radius returns a BlastRadiusReport."""
    report = await ekg.calculate_blast_radius("payments-db")
    assert isinstance(report, BlastRadiusReport)
    assert report.target_service == "payments-db"


@pytest.mark.asyncio
async def test_blast_radius_tier1_impact(ekg):
    """payments-db failure should impact tier-1 services (payments-service, checkout-service)."""
    report = await ekg.calculate_blast_radius("payments-db")
    assert report.tier1_impact
    affected = set(report.affected_services)
    assert "payments-service" in affected or "checkout-service" in affected


@pytest.mark.asyncio
async def test_blast_radius_leaf_node(ekg):
    """analytics-service (tier-3, no upstream dependents) has low blast radius."""
    report = await ekg.calculate_blast_radius("analytics-service")
    assert len(report.affected_services) == 0
    assert not report.tier1_impact
    assert report.recommendation in ("auto_execute", "require_approval")


@pytest.mark.asyncio
async def test_blast_radius_risk_score_bounded(ekg):
    """risk_score is always in [0.0, 1.0]."""
    for svc in ["payments-db", "analytics-service", "checkout-service"]:
        report = await ekg.calculate_blast_radius(svc)
        assert 0.0 <= report.risk_score <= 1.0, f"risk_score out of bounds for {svc}"


@pytest.mark.asyncio
async def test_blast_radius_recommendation_valid(ekg):
    """recommendation is one of the three valid values."""
    report = await ekg.calculate_blast_radius("payments-db")
    assert report.recommendation in ("auto_execute", "require_approval", "block")


@pytest.mark.asyncio
async def test_blast_radius_to_markdown(ekg):
    """to_markdown() includes the target service name."""
    report = await ekg.calculate_blast_radius("payments-db")
    md = report.to_markdown()
    assert "payments-db" in md
    assert isinstance(md, str)


# ---------------------------------------------------------------------------
# Tests: update_node_health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_node_health_critical(ekg):
    """update_node_health changes health_status to the given value."""
    await ekg.update_node_health("payments-service", HealthStatus.CRITICAL)
    attrs = ekg._graph.nodes.get("payments-service", {})
    assert attrs.get("health_status") == HealthStatus.CRITICAL.value


@pytest.mark.asyncio
async def test_update_node_health_unknown_node(ekg):
    """update_node_health on a missing node does not raise."""
    await ekg.update_node_health("nonexistent-svc", HealthStatus.DEGRADED)


@pytest.mark.asyncio
async def test_update_node_health_healthy(ekg):
    """Can restore health back to HEALTHY."""
    await ekg.update_node_health("payments-db", HealthStatus.CRITICAL)
    await ekg.update_node_health("payments-db", HealthStatus.HEALTHY)
    attrs = ekg._graph.nodes.get("payments-db", {})
    assert attrs.get("health_status") == HealthStatus.HEALTHY.value


# ---------------------------------------------------------------------------
# Tests: failure correlations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_failure_correlations_found(ekg):
    """get_failure_correlations returns patterns containing the service."""
    corrs = await ekg.get_failure_correlations("payments-db")
    assert len(corrs) >= 1
    assert any("payments-db" in c.get("services", []) for c in corrs)


@pytest.mark.asyncio
async def test_get_failure_correlations_not_found(ekg):
    """analytics-service has no known correlations."""
    corrs = await ekg.get_failure_correlations("analytics-service")
    assert corrs == []


# ---------------------------------------------------------------------------
# Tests: all_service_names
# ---------------------------------------------------------------------------

def test_all_service_names_after_load(ekg):
    """all_service_names() lists all loaded nodes synchronously."""
    # ekg fixture is async but the property is sync — access via the loop
    names = ekg.all_service_names()
    assert isinstance(names, list)
    # At least our 5 fixture nodes should be there
    assert len(names) == 5
    assert "payments-db" in names
