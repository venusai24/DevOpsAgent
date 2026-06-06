"""
airs_v2.context — Infrastructure Context Layer (Stage 2)
=========================================================

This package provides the **Context Layer** for AIRS v2: a typed,
traversal-aware infrastructure dependency graph exposed exclusively
via a Model Context Protocol (MCP) server.

The reasoning engine must query all infrastructure state through the MCP
tools defined here. This ensures the AI's context window is strictly
bounded to the topological blast radius of the incident, preventing
hallucinated connections and unnecessary token consumption.

Public surface
--------------
    ContextGraph          — NetworkX DiGraph with typed Pydantic nodes/edges
    ContextMCPServer      — FastMCP server wrapping ContextGraph as tools
    ContextMCPClient      — Async stdio client for calling the MCP server

MCP Tools (exposed by ContextMCPServer)
----------------------------------------
    get_blast_radius        — All nodes within depth-2 of a focal service
    get_subgraph            — Node + edge list bounded to blast radius
    get_node_health         — Point-in-time health snapshot of one node
    get_metrics_snapshot    — Connection saturation, error rate, replica state
    get_failure_correlations— Historical co-failure partners
    update_node_health      — Mutate a node's health status in real time

Design constraints
------------------
* Blast radius depth is hardcoded to 2 (Stage 2 invariant).
* MetricsSnapshot carries ``stale=True`` until live API data is connected
  (Stage 3) so the reasoning engine can apply lower confidence weights.
* Zero dependency on `agent/` internals — this package is standalone.
"""

from airs_v2.context.graph import ContextGraph
from airs_v2.context.mcp_server import build_mcp_server

__all__ = ["ContextGraph", "build_mcp_server"]
