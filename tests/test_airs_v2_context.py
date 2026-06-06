"""
tests/test_airs_v2_context.py
==============================

Integration tests for Stage 2 — Context Layer (ContextGraph + MCP Server).

Test structure
--------------
TestContextGraph           — Direct ContextGraph unit tests: loading, node/edge
                             counts, blast radius, subgraph bounding, health
                             mutation, metrics, correlations.
TestBlastRadiusBounding    — Invariant: subgraph never contains nodes outside depth-2.
TestHealthPropagation      — update_health mutation reflected in subsequent queries.
TestMCPToolsDirect         — Call FastMCP tools in-process (no subprocess).
TestMCPClientIntegration   — Full stdio subprocess round-trip via ContextMCPClient.
TestAlertScenarios         — 3 real alert scenarios: verify MCP returns ONLY
                             topologically relevant nodes.

Alert scenario ground truth (from topology_fixtures.json)
----------------------------------------------------------
Scenario A: payments-db CRITICAL
  Blast radius upstream (depth ≤ 2):
    depth-1: payments-service, payment-processor
    depth-2: order-processing-api (via payments-service),
             api-gateway (via payments-service)
  Expected in subgraph: payments-db + all above + edges
  Expected NOT in subgraph: auth-service, inventory-service, thirdparty-sso, ...

Scenario B: redis-user-cache CRITICAL
  Upstream: user-profile-service (depth-1), payments-service (depth-1)
            api-gateway (depth-2, via payments-service),
            order-processing-api (depth-2, via payments-service)

Scenario C: thirdparty-sso DEGRADED
  Upstream: auth-service (depth-1), authentication-service (depth-1)
            api-gateway (depth-2, via auth-service)

All tests run in the genai conda environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from airs_v2.context.graph import (
    ContextGraph,
    BLAST_RADIUS_DEPTH,
    NodeSnapshot,
    MetricsSnapshot,
    ContextSubgraph,
    BlastRadiusResult,
    CorrelationRecord,
)
from airs_v2.context.mcp_server import build_mcp_server, reset_graph
from airs_v2.context.client import ContextMCPClient, CallError

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

FIXTURES_PATH = (
    Path(__file__).resolve().parents[1]
    / "mock_enterprise"
    / "topology_fixtures.json"
)

ALL_KNOWN_SERVICES = {
    "api-gateway", "payments-service", "payment-processor",
    "auth-service", "authentication-service", "inventory-service",
    "user-profile-service", "order-processing-api",
    "payments-db", "auth-db", "inventory-db", "user-db",
    "redis-user-cache", "redis-cache", "thirdparty-sso",
}

# Services that must NOT appear in payments-db blast radius
PAYMENTS_DB_EXCLUDED = {
    "auth-service", "authentication-service", "auth-db",
    "inventory-service", "inventory-db", "redis-cache",
    "thirdparty-sso", "user-db",
}


@pytest.fixture(scope="module")
def graph() -> ContextGraph:
    """Module-scoped ContextGraph seeded from the real fixture file."""
    return ContextGraph(fixtures_path=FIXTURES_PATH)


@pytest.fixture(scope="function")
def isolated_graph() -> ContextGraph:
    """Function-scoped graph for tests that mutate health state."""
    return ContextGraph(fixtures_path=FIXTURES_PATH)


# ---------------------------------------------------------------------------
# TestContextGraph
# ---------------------------------------------------------------------------

class TestContextGraph:
    """Unit tests for ContextGraph construction and query methods."""

    def test_graph_loads_all_nodes(self, graph):
        names = set(graph.node_names())
        assert ALL_KNOWN_SERVICES.issubset(names)

    def test_graph_node_count(self, graph):
        assert graph._g.number_of_nodes() >= 15

    def test_graph_has_dependency_edges(self, graph):
        assert graph._g.number_of_edges() > 0

    def test_payments_service_depends_on_payments_db(self, graph):
        assert graph._g.has_edge("payments-service", "payments-db")

    def test_api_gateway_depends_on_payments_service(self, graph):
        assert graph._g.has_edge("api-gateway", "payments-service")

    def test_get_node_returns_snapshot(self, graph):
        snap = graph.get_node("payments-service")
        assert isinstance(snap, NodeSnapshot)
        assert snap.name == "payments-service"
        assert snap.tier == 1

    def test_get_node_unknown_returns_none(self, graph):
        assert graph.get_node("nonexistent-svc") is None

    def test_get_node_health_status_correct(self, graph):
        snap = graph.get_node("payments-db")
        assert snap is not None
        assert snap.health_status == "critical"

    def test_stats_structure(self, graph):
        s = graph.stats
        assert s["node_count"] >= 15
        assert s["edge_count"] > 0
        assert s["loaded"] is True
        assert s["blast_radius_depth"] == BLAST_RADIUS_DEPTH

    def test_metrics_snapshot_returns_object(self, graph):
        m = graph.get_metrics_snapshot("payments-db")
        assert isinstance(m, MetricsSnapshot)
        assert m.service == "payments-db"

    def test_metrics_stale_true_in_stage2(self, graph):
        for svc in ["payments-service", "auth-db", "redis-user-cache"]:
            m = graph.get_metrics_snapshot(svc)
            assert m is not None
            assert m.stale is True, f"Expected stale=True for {svc}"

    def test_metrics_db_saturation_critical(self, graph):
        m = graph.get_metrics_snapshot("payments-db")
        assert m is not None
        assert m.connection_saturation == pytest.approx(1.0)

    def test_metrics_auth_db_saturation_low(self, graph):
        m = graph.get_metrics_snapshot("auth-db")
        assert m is not None
        assert m.connection_saturation == pytest.approx(0.225, abs=0.01)

    def test_metrics_unknown_service_returns_none(self, graph):
        assert graph.get_metrics_snapshot("no-such-service") is None

    def test_correlations_payments_service(self, graph):
        records = graph.get_failure_correlations("payments-service")
        assert len(records) > 0
        partners = {r.partner_service for r in records}
        assert "api-gateway" in partners or "order-processing-api" in partners

    def test_correlations_returns_list_of_records(self, graph):
        records = graph.get_failure_correlations("payments-service")
        for r in records:
            assert isinstance(r, CorrelationRecord)
            assert isinstance(r.historical_incident_ids, list)

    def test_correlations_no_match_returns_empty(self, graph):
        records = graph.get_failure_correlations("auth-db")
        assert isinstance(records, list)


# ---------------------------------------------------------------------------
# TestBlastRadiusBounding
# ---------------------------------------------------------------------------

class TestBlastRadiusBounding:
    """
    Key correctness invariant: subgraph() must NEVER return a node that is
    more than BLAST_RADIUS_DEPTH hops from the focal service.
    """

    def _hop_distance(self, graph: ContextGraph, source: str, target: str) -> int:
        import networkx as nx
        undirected = graph._g.to_undirected()
        try:
            return nx.shortest_path_length(undirected, source, target)
        except Exception:
            return 999

    def test_subgraph_payments_db_bounding(self, graph):
        sg = graph.subgraph("payments-db")
        for node in sg.nodes:
            if node.name == "payments-db":
                continue
            dist = self._hop_distance(graph, "payments-db", node.name)
            assert dist <= BLAST_RADIUS_DEPTH, (
                f"Node '{node.name}' is {dist} hops from payments-db "
                f"but BLAST_RADIUS_DEPTH={BLAST_RADIUS_DEPTH}"
            )

    def test_subgraph_redis_cache_bounding(self, graph):
        sg = graph.subgraph("redis-user-cache")
        for node in sg.nodes:
            if node.name == "redis-user-cache":
                continue
            dist = self._hop_distance(graph, "redis-user-cache", node.name)
            assert dist <= BLAST_RADIUS_DEPTH

    def test_subgraph_thirdparty_sso_bounding(self, graph):
        sg = graph.subgraph("thirdparty-sso")
        for node in sg.nodes:
            if node.name == "thirdparty-sso":
                continue
            dist = self._hop_distance(graph, "thirdparty-sso", node.name)
            assert dist <= BLAST_RADIUS_DEPTH

    def test_excluded_nodes_absent_payments_db(self, graph):
        sg = graph.subgraph("payments-db")
        node_names = {n.name for n in sg.nodes}
        for excluded in PAYMENTS_DB_EXCLUDED:
            assert excluded not in node_names, (
                f"'{excluded}' should NOT be in payments-db subgraph "
                f"(blast radius depth={BLAST_RADIUS_DEPTH})"
            )

    def test_edges_stay_within_subgraph(self, graph):
        sg = graph.subgraph("payments-service")
        in_scope = {n.name for n in sg.nodes}
        for edge in sg.edges:
            assert edge.source in in_scope, f"Edge source '{edge.source}' out of scope"
            assert edge.target in in_scope, f"Edge target '{edge.target}' out of scope"

    def test_blast_radius_depth_constant(self):
        assert BLAST_RADIUS_DEPTH == 2

    def test_blast_radius_result_structure(self, graph):
        br = graph.blast_radius("payments-service")
        assert isinstance(br, BlastRadiusResult)
        assert br.focus_service == "payments-service"
        assert isinstance(br.affected_services, list)
        assert isinstance(br.tier1_services, list)
        assert isinstance(br.tier1_impact, bool)
        assert 0.0 <= br.risk_score <= 1.0
        assert isinstance(br.propagation_paths, list)
        assert isinstance(br.on_call_contacts, list)


# ---------------------------------------------------------------------------
# TestHealthPropagation
# ---------------------------------------------------------------------------

class TestHealthPropagation:
    """Verifies that update_health mutations are immediately reflected."""

    def test_update_health_changes_node_snapshot(self, isolated_graph):
        isolated_graph.update_health("auth-service", "degraded")
        snap = isolated_graph.get_node("auth-service")
        assert snap is not None
        assert snap.health_status == "degraded"

    def test_update_health_reflected_in_subgraph(self, isolated_graph):
        isolated_graph.update_health("api-gateway", "critical")
        sg = isolated_graph.subgraph("payments-service")
        node_map = {n.name: n for n in sg.nodes}
        if "api-gateway" in node_map:
            assert node_map["api-gateway"].health_status == "critical"

    def test_update_health_to_healthy(self, isolated_graph):
        isolated_graph.update_health("payments-db", "healthy")
        snap = isolated_graph.get_node("payments-db")
        assert snap is not None
        assert snap.health_status == "healthy"

    def test_update_health_returns_false_for_unknown(self, isolated_graph):
        ok = isolated_graph.update_health("phantom-service", "critical")
        assert ok is False

    def test_update_health_invalid_status_raises(self, isolated_graph):
        with pytest.raises(ValueError, match="Invalid health status"):
            isolated_graph.update_health("payments-service", "on_fire")

    def test_multiple_updates_tracked(self, isolated_graph):
        isolated_graph.update_health("payments-db", "degraded")
        isolated_graph.update_health("payments-db", "critical")
        snap = isolated_graph.get_node("payments-db")
        assert snap.health_status == "critical"


# ---------------------------------------------------------------------------
# TestMCPToolsDirect
# ---------------------------------------------------------------------------

class TestMCPToolsDirect:
    """
    Call FastMCP tools in-process without a subprocess.

    FastMCP.call_tool(name, arguments) → tuple(list[TextContent], is_error: bool)
    Extract text via: content_list, _is_error = result; content_list[0].text
    """

    @staticmethod
    def _text(result) -> str:
        """Extract text from FastMCP.call_tool() return tuple."""
        content_list, _is_error = result
        return content_list[0].text

    @pytest.fixture(autouse=True)
    def inject_graph(self, isolated_graph):
        self.mcp = build_mcp_server(graph=isolated_graph)
        self.graph = isolated_graph
        yield
        reset_graph(None)

    @pytest.mark.asyncio
    async def test_get_blast_radius_returns_json(self):
        result = await self.mcp.call_tool("get_blast_radius", {"service": "payments-db"})
        data = json.loads(self._text(result))
        assert "affected_services" in data
        assert "tier1_services" in data
        assert "risk_score" in data

    @pytest.mark.asyncio
    async def test_get_blast_radius_unknown_service_returns_error(self):
        result = await self.mcp.call_tool("get_blast_radius", {"service": "no-such-service"})
        data = json.loads(self._text(result))
        assert "error" in data
        assert "known_services" in data

    @pytest.mark.asyncio
    async def test_get_subgraph_returns_nodes_and_edges(self):
        result = await self.mcp.call_tool("get_subgraph", {"service": "payments-service"})
        data = json.loads(self._text(result))
        assert "nodes" in data
        assert "edges" in data
        assert "markdown_summary" in data
        assert len(data["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_get_subgraph_excluded_nodes_absent(self):
        result = await self.mcp.call_tool("get_subgraph", {"service": "payments-db"})
        data = json.loads(self._text(result))
        node_names = {n["name"] for n in data["nodes"]}
        for excluded in PAYMENTS_DB_EXCLUDED:
            assert excluded not in node_names, (
                f"Node '{excluded}' must NOT appear in payments-db MCP subgraph"
            )

    @pytest.mark.asyncio
    async def test_get_node_health_payments_db(self):
        result = await self.mcp.call_tool("get_node_health", {"service": "payments-db"})
        data = json.loads(self._text(result))
        assert data["name"] == "payments-db"
        assert data["health_status"] == "critical"
        assert data["tier"] == 1

    @pytest.mark.asyncio
    async def test_get_node_health_unknown_returns_error(self):
        result = await self.mcp.call_tool("get_node_health", {"service": "ghost-service"})
        data = json.loads(self._text(result))
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_metrics_snapshot_stale_flag(self):
        result = await self.mcp.call_tool("get_metrics_snapshot", {"service": "payments-db"})
        data = json.loads(self._text(result))
        assert data["stale"] is True

    @pytest.mark.asyncio
    async def test_get_metrics_snapshot_saturation_critical(self):
        result = await self.mcp.call_tool("get_metrics_snapshot", {"service": "payments-db"})
        data = json.loads(self._text(result))
        assert data["connection_saturation"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_get_failure_correlations_payments_service(self):
        result = await self.mcp.call_tool(
            "get_failure_correlations", {"service": "payments-service"}
        )
        records = json.loads(self._text(result))
        assert isinstance(records, list)
        assert len(records) > 0
        partners = {r["partner_service"] for r in records}
        assert "api-gateway" in partners or "order-processing-api" in partners

    @pytest.mark.asyncio
    async def test_get_failure_correlations_no_match(self):
        result = await self.mcp.call_tool(
            "get_failure_correlations", {"service": "redis-cache"}
        )
        records = json.loads(self._text(result))
        assert isinstance(records, list)

    @pytest.mark.asyncio
    async def test_update_node_health_success(self):
        result = await self.mcp.call_tool(
            "update_node_health", {"service": "auth-service", "status": "degraded"}
        )
        data = json.loads(self._text(result))
        assert data["updated"] is True
        assert data["new_status"] == "degraded"

    @pytest.mark.asyncio
    async def test_update_node_health_reflected_in_get_node(self):
        await self.mcp.call_tool(
            "update_node_health", {"service": "payments-db", "status": "healthy"}
        )
        result = await self.mcp.call_tool("get_node_health", {"service": "payments-db"})
        data = json.loads(self._text(result))
        assert data["health_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_update_node_health_invalid_status(self):
        result = await self.mcp.call_tool(
            "update_node_health", {"service": "payments-service", "status": "exploding"}
        )
        data = json.loads(self._text(result))
        assert "error" in data

    @pytest.mark.asyncio
    async def test_subgraph_markdown_contains_focus(self):
        result = await self.mcp.call_tool("get_subgraph", {"service": "payments-service"})
        data = json.loads(self._text(result))
        assert "payments-service" in data["markdown_summary"]

    @pytest.mark.asyncio
    async def test_subgraph_tier1_nodes_reported(self):
        result = await self.mcp.call_tool("get_subgraph", {"service": "payments-db"})
        data = json.loads(self._text(result))
        assert len(data["tier1_node_names"]) > 0


# ---------------------------------------------------------------------------
# TestMCPClientIntegration
# ---------------------------------------------------------------------------

class TestMCPClientIntegration:
    """
    Full stdio subprocess round-trip via ContextMCPClient.

    The client spawns the server as a real subprocess; validates the complete
    transport stack (stdin/stdout pipes, JSON-RPC framing, MCP initialize
    handshake, anyio cancel-scope lifecycle).
    """

    @pytest.mark.asyncio
    async def test_client_lists_all_six_tools(self):
        async with ContextMCPClient() as client:
            tools = await client.list_tools()
        expected = {
            "get_blast_radius", "get_subgraph", "get_node_health",
            "get_metrics_snapshot", "get_failure_correlations",
            "update_node_health",
        }
        assert expected.issubset(set(tools)), f"Missing tools: {expected - set(tools)}"

    @pytest.mark.asyncio
    async def test_client_get_blast_radius(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="payments-db")
        assert data["focus_service"] == "payments-db"
        assert isinstance(data["affected_services"], list)
        assert "payments-service" in data["affected_services"]

    @pytest.mark.asyncio
    async def test_client_get_subgraph_payments_service(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_subgraph", service="payments-service")
        assert "nodes" in data
        assert "edges" in data
        node_names = {n["name"] for n in data["nodes"]}
        assert "payments-service" in node_names
        assert "payments-db" in node_names  # depth-1 dependency

    @pytest.mark.asyncio
    async def test_client_get_node_health_known(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_node_health", service="auth-db")
        assert data["name"] == "auth-db"
        assert data["health_status"] in {"healthy", "degraded", "critical", "unknown"}

    @pytest.mark.asyncio
    async def test_client_get_node_health_unknown_raises_call_error(self):
        """CallError must be raised when server returns an error envelope."""
        raised = None
        async with ContextMCPClient() as client:
            try:
                await client.call("get_node_health", service="phantom-99")
            except CallError as exc:
                raised = exc
        assert raised is not None, "Expected CallError was not raised"
        assert "phantom-99" in str(raised) or "not found" in str(raised).lower()

    @pytest.mark.asyncio
    async def test_client_metrics_stale_flag(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_metrics_snapshot", service="payments-db")
        assert data["stale"] is True

    @pytest.mark.asyncio
    async def test_client_correlations_payments_service(self):
        async with ContextMCPClient() as client:
            records = await client.call("get_failure_correlations", service="payments-service")
        assert len(records) > 0

    @pytest.mark.asyncio
    async def test_client_update_health_and_verify(self):
        async with ContextMCPClient() as client:
            ack = await client.call(
                "update_node_health", service="api-gateway", status="degraded"
            )
            assert ack["updated"] is True
            snap = await client.call("get_node_health", service="api-gateway")
        assert snap["health_status"] == "degraded"


# ---------------------------------------------------------------------------
# TestAlertScenarios
# ---------------------------------------------------------------------------

class TestAlertScenarios:
    """
    Simulates real P0/P1 alerts and verifies the MCP server returns ONLY
    topologically relevant data.

    Primary correctness invariant: the reasoning engine receives a bounded
    context window. No node outside the blast radius may appear.
    """

    # ----------------------------------------------------------------
    # Scenario A: payments-db CRITICAL (connection pool exhausted)
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_scenario_a_blast_radius_includes_payment_consumers(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="payments-db")
        affected = set(data["affected_services"])
        assert "payments-service" in affected
        assert "payment-processor" in affected

    @pytest.mark.asyncio
    async def test_scenario_a_blast_radius_excludes_unrelated_services(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="payments-db")
        affected = set(data["affected_services"])
        for excluded in PAYMENTS_DB_EXCLUDED:
            assert excluded not in affected, (
                f"INVARIANT VIOLATION: '{excluded}' is in payments-db blast radius "
                f"but is more than {BLAST_RADIUS_DEPTH} hops away"
            )

    @pytest.mark.asyncio
    async def test_scenario_a_tier1_impact_detected(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="payments-db")
        assert data["tier1_impact"] is True

    @pytest.mark.asyncio
    async def test_scenario_a_subgraph_bounded(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_subgraph", service="payments-db")
        node_names = {n["name"] for n in data["nodes"]}
        for excluded in PAYMENTS_DB_EXCLUDED:
            assert excluded not in node_names, (
                f"INVARIANT VIOLATION: '{excluded}' in payments-db MCP subgraph"
            )

    @pytest.mark.asyncio
    async def test_scenario_a_metrics_show_saturation(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_metrics_snapshot", service="payments-db")
        assert data["connection_saturation"] == pytest.approx(1.0)
        assert data["stale"] is True

    @pytest.mark.asyncio
    async def test_scenario_a_health_update_then_requery(self):
        async with ContextMCPClient() as client:
            await client.call(
                "update_node_health", service="payments-db", status="healthy"
            )
            snap = await client.call("get_node_health", service="payments-db")
        assert snap["health_status"] == "healthy"

    # ----------------------------------------------------------------
    # Scenario B: redis-user-cache CRITICAL (OOM)
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_scenario_b_blast_radius_includes_cache_consumers(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="redis-user-cache")
        affected = set(data["affected_services"])
        assert "user-profile-service" in affected
        assert "payments-service" in affected

    @pytest.mark.asyncio
    async def test_scenario_b_blast_radius_excludes_inventory_services(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="redis-user-cache")
        affected = set(data["affected_services"])
        assert "inventory-db" not in affected
        assert "auth-db" not in affected

    @pytest.mark.asyncio
    async def test_scenario_b_correlations_known(self):
        async with ContextMCPClient() as client:
            records = await client.call(
                "get_failure_correlations", service="user-profile-service"
            )
        partners = {r["partner_service"] for r in records}
        assert "payments-service" in partners

    # ----------------------------------------------------------------
    # Scenario C: thirdparty-sso DEGRADED
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_scenario_c_blast_radius_includes_auth_services(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="thirdparty-sso")
        affected = set(data["affected_services"])
        assert "auth-service" in affected
        assert "authentication-service" in affected

    @pytest.mark.asyncio
    async def test_scenario_c_api_gateway_in_blast_radius(self):
        """api-gateway → auth-service → thirdparty-sso: depth-2."""
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="thirdparty-sso")
        affected = set(data["affected_services"])
        assert "api-gateway" in affected, (
            "api-gateway (depth-2 via auth-service) must be in thirdparty-sso blast radius"
        )

    @pytest.mark.asyncio
    async def test_scenario_c_payments_db_excluded(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_blast_radius", service="thirdparty-sso")
        affected = set(data["affected_services"])
        assert "payments-db" not in affected
        assert "inventory-db" not in affected

    @pytest.mark.asyncio
    async def test_scenario_c_subgraph_markdown_mentions_tier1(self):
        async with ContextMCPClient() as client:
            data = await client.call("get_subgraph", service="thirdparty-sso")
        assert "api-gateway" in data["tier1_node_names"]

    @pytest.mark.asyncio
    async def test_scenario_c_update_and_blast_radius_unchanged(self):
        """Health mutation must not alter graph topology or blast radius set."""
        async with ContextMCPClient() as client:
            await client.call(
                "update_node_health", service="thirdparty-sso", status="critical"
            )
            data = await client.call("get_blast_radius", service="thirdparty-sso")
        affected = set(data["affected_services"])
        assert "auth-service" in affected
        assert "api-gateway" in affected
