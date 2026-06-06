"""
airs_v2/context/mcp_server.py
==============================

ContextMCPServer — MCP Server for Infrastructure Context
---------------------------------------------------------

Wraps a ``ContextGraph`` singleton and exposes its query methods as
Model Context Protocol (MCP) tools via ``FastMCP``.

The reasoning engine is the **only** consumer of infrastructure state —
it must call these tools rather than querying the graph directly. This
architectural boundary is what keeps the AI's context window bounded to the
topological blast radius of the incident.

Transport
---------
``stdio`` — the server is designed to be spawned as a subprocess and
connected via stdin/stdout pipes using ``mcp.client.stdio.stdio_client``.
In integration tests, the server module is run directly as a subprocess::

    python -m airs_v2.context.mcp_server

For in-process unit testing of tool logic, use ``build_mcp_server()`` and
call ``mcp.call_tool(name, args)`` directly.

Tools exposed
-------------
+---------------------------+----------------------------------------+
| Tool                      | Description                            |
+===========================+========================================+
| get_blast_radius          | Full blast radius analysis (depth=2)   |
| get_subgraph              | Bounded node+edge set + markdown       |
| get_node_health           | Point-in-time health of one node       |
| get_metrics_snapshot      | Connection saturation, error rate      |
| get_failure_correlations  | Historical co-failure partners         |
| update_node_health        | Mutate live health status              |
+---------------------------+----------------------------------------+

All tools return JSON-serialisable strings.  Errors are surfaced as
structured ``{"error": "...", "service": "..."}`` dicts so the reasoning
engine can self-correct on bad input.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from airs_v2.context.graph import ContextGraph, BLAST_RADIUS_DEPTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton graph — one instance shared across all tool calls in a session
# ---------------------------------------------------------------------------
_GRAPH: Optional[ContextGraph] = None


def _graph() -> ContextGraph:
    """Lazy-init the ContextGraph singleton."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = ContextGraph()
    return _GRAPH


def reset_graph(graph: Optional[ContextGraph] = None) -> None:
    """
    Replace the singleton graph.  Used in tests to inject a pre-seeded
    or isolated graph without touching the module-level singleton.
    """
    global _GRAPH
    _GRAPH = graph


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------

def build_mcp_server(graph: Optional[ContextGraph] = None) -> FastMCP:
    """
    Construct and return a configured FastMCP server instance.

    Parameters
    ----------
    graph:
        Optional pre-built ContextGraph to use instead of the singleton.
        Pass a custom graph in tests for full isolation.

    Returns
    -------
    FastMCP
        A configured server ready for ``run_stdio_async()`` or direct
        ``call_tool()`` invocation in tests.
    """
    if graph is not None:
        reset_graph(graph)

    mcp = FastMCP(
        name="airs-context-server",
        instructions=(
            "Infrastructure Context MCP Server for AIRS v2.\n\n"
            "IMPORTANT: All infrastructure topology queries MUST go through "
            "these tools. Never infer service dependencies from log text alone. "
            "Always call get_blast_radius first to bound the scope of analysis, "
            "then use get_subgraph to get the full bounded context. "
            "Nodes absent from the subgraph are outside your context window — "
            "do not speculate about them."
        ),
    )

    # ----------------------------------------------------------------
    # Tool: get_blast_radius
    # ----------------------------------------------------------------

    @mcp.tool()
    async def get_blast_radius(service: str) -> str:
        """
        Compute the blast radius of a service at depth=2.

        Returns the set of infrastructure nodes that are within 2 dependency
        hops of the focal service — both upstream callers (services that
        depend on it) and downstream dependencies (things it calls).

        Use this tool FIRST when an alert arrives to understand the scope of
        impact before fetching logs or metrics.

        Args:
            service: Canonical service name (e.g. 'payments-service',
                     'payments-db', 'redis-user-cache').

        Returns:
            JSON with: focus_service, affected_services (list),
            tier1_services (list), tier1_impact (bool), risk_score (0–1),
            propagation_paths (list of paths), on_call_contacts (list).
        """
        g = _graph()
        if not g._g.has_node(service):
            return json.dumps({
                "error": f"Service '{service}' not found in topology graph.",
                "known_services": g.node_names()[:20],
            })
        result = g.blast_radius(service)
        return result.model_dump_json()

    # ----------------------------------------------------------------
    # Tool: get_subgraph
    # ----------------------------------------------------------------

    @mcp.tool()
    async def get_subgraph(service: str) -> str:
        """
        Retrieve the full bounded topology subgraph for a focal service.

        Returns all nodes AND edges within the depth-2 blast radius, plus a
        markdown summary. This is your complete context window for topology
        analysis — no nodes outside this set should be considered.

        Args:
            service: Canonical service name (e.g. 'payments-service').

        Returns:
            JSON with: focus_service, nodes (list of NodeSnapshot),
            edges (list of EdgeSnapshot), affected_count (int),
            tier1_node_names (list), markdown_summary (str).
        """
        g = _graph()
        if not g._g.has_node(service):
            return json.dumps({
                "error": f"Service '{service}' not found in topology graph.",
                "known_services": g.node_names()[:20],
            })
        sg = g.subgraph(service)
        data = sg.model_dump()
        data["markdown_summary"] = sg.to_markdown()
        return json.dumps(data, default=str)

    # ----------------------------------------------------------------
    # Tool: get_node_health
    # ----------------------------------------------------------------

    @mcp.tool()
    async def get_node_health(service: str) -> str:
        """
        Get the current health status and metadata of a single node.

        Use this to check the health of a specific node before deciding on a
        remediation action. Health status values: healthy | degraded |
        critical | unknown.

        Args:
            service: Canonical service name.

        Returns:
            JSON NodeSnapshot with: name, node_type, tier, health_status,
            owner, namespace, on_call, metadata, captured_at.
        """
        g = _graph()
        snap = g.get_node(service)
        if snap is None:
            return json.dumps({
                "error": f"Service '{service}' not found.",
                "service": service,
            })
        return snap.model_dump_json()

    # ----------------------------------------------------------------
    # Tool: get_metrics_snapshot
    # ----------------------------------------------------------------

    @mcp.tool()
    async def get_metrics_snapshot(service: str) -> str:
        """
        Retrieve infrastructure metrics for a single node.

        Returns connection pool saturation, error rate, replica counts, and
        memory usage. The 'stale' flag indicates whether data comes from the
        static topology fixture (stale=true) or a live API query (stale=false).

        IMPORTANT: When stale=true, treat these metrics as indicative rather
        than definitive. Do not use stale metrics as the sole evidence for a
        root cause determination.

        Args:
            service: Canonical service name.

        Returns:
            JSON MetricsSnapshot with: service, connection_saturation (0–1),
            error_rate_pct, replica_desired, replica_ready, memory_usage_pct,
            stale (bool), captured_at.
        """
        g = _graph()
        snap = g.get_metrics_snapshot(service)
        if snap is None:
            return json.dumps({
                "error": f"Service '{service}' not found.",
                "service": service,
            })
        return snap.model_dump_json()

    # ----------------------------------------------------------------
    # Tool: get_failure_correlations
    # ----------------------------------------------------------------

    @mcp.tool()
    async def get_failure_correlations(service: str) -> str:
        """
        Look up historical co-failure partners for a service.

        Returns services that have been observed to fail together with the
        focal service in past incidents, including a description of the failure
        cascade mechanism and references to historical incident IDs.

        Use this tool after get_blast_radius to understand which correlated
        failures are most likely to co-occur, prioritising your investigation.

        Args:
            service: Canonical service name.

        Returns:
            JSON array of CorrelationRecord objects, each with:
            partner_service, description, historical_incident_ids.
            Empty array if no correlations are known.
        """
        g = _graph()
        records = g.get_failure_correlations(service)
        return json.dumps([r.model_dump() for r in records])

    # ----------------------------------------------------------------
    # Tool: update_node_health
    # ----------------------------------------------------------------

    @mcp.tool()
    async def update_node_health(service: str, status: str) -> str:
        """
        Update the live health status of an infrastructure node.

        Call this when the perception layer or a monitoring tool has confirmed
        a health state change. Subsequent calls to get_subgraph or
        get_blast_radius will reflect the updated status.

        Valid status values: healthy | degraded | critical | unknown

        Args:
            service: Canonical service name to update.
            status:  New health status (healthy | degraded | critical | unknown).

        Returns:
            Confirmation JSON with the updated node snapshot, or an error if
            the service is not found or the status is invalid.
        """
        g = _graph()
        try:
            ok = g.update_health(service, status)
        except ValueError as exc:
            return json.dumps({"error": str(exc), "service": service})
        if not ok:
            return json.dumps({
                "error": f"Service '{service}' not found.",
                "service": service,
            })
        snap = g.get_node(service)
        return json.dumps({
            "updated": True,
            "service": service,
            "new_status": status,
            "node": snap.model_dump() if snap else None,
        })

    return mcp


# ---------------------------------------------------------------------------
# Entry point — runs as stdio MCP server when invoked directly
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the ContextMCPServer over stdio transport."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = build_mcp_server()
    import asyncio
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
