"""
Tool Registry — Module 1.6.

Maps evidence gaps (missing_mass categories) to concrete MCP tool specs.
Loads tool definitions from the C2 collection (tool_selection) or falls back
to the hardcoded manifest derived from MCPTools.md.

Phase 1: Hardcoded manifest. Phase 2: Loads from Qdrant C2 collection.
"""
from __future__ import annotations

import logging
from typing import Optional

from airs.models.intents import SignalType, ToolSpec, ToolTier

log = logging.getLogger(__name__)


# ─── Hardcoded Tool Manifest (from MCPTools.md) ───────────────────────────────
# Each entry defines one available MCP tool.
# This is the Phase 1 fallback; Phase 2 loads from the Qdrant C2 collection.

_TOOL_MANIFEST: list[dict] = [
    # ── Prometheus (Metrics) ──────────────────────────────────────────────────
    {
        "tool_name": "execute_query",
        "mcp_server_id": "prometheus",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.METRICS,
        "idempotent": True,
        "expected_latency_seconds": 5,
        "evidence_categories": ["metrics_current_state", "error_rate", "cpu_utilization", "memory_usage"],
        "description": "Execute PromQL instant query",
    },
    {
        "tool_name": "execute_range_query",
        "mcp_server_id": "prometheus",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.METRICS,
        "idempotent": True,
        "expected_latency_seconds": 8,
        "evidence_categories": ["metrics_trend", "latency_p99", "throughput_trend"],
        "description": "Execute PromQL range query over a time window",
    },
    {
        "tool_name": "get_metric_metadata",
        "mcp_server_id": "prometheus",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.METRICS,
        "idempotent": True,
        "expected_latency_seconds": 3,
        "evidence_categories": ["metric_existence", "metric_labels"],
        "description": "Retrieve metric labels and metadata",
    },
    {
        "tool_name": "list_metrics",
        "mcp_server_id": "prometheus",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.METRICS,
        "idempotent": True,
        "expected_latency_seconds": 3,
        "evidence_categories": ["available_signals"],
        "description": "List available Prometheus metrics",
    },
    # ── Loki (Logs) ───────────────────────────────────────────────────────────
    {
        "tool_name": "loki_query",
        "mcp_server_id": "loki",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.LOGS,
        "idempotent": True,
        "expected_latency_seconds": 10,
        "evidence_categories": ["application_logs", "error_logs", "service_logs"],
        "description": "Query Grafana Loki log data with LogQL",
    },
    # ── OpenSearch (Logs) ─────────────────────────────────────────────────────
    {
        "tool_name": "SearchIndexTool",
        "mcp_server_id": "opensearch",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.LOGS,
        "idempotent": True,
        "expected_latency_seconds": 8,
        "evidence_categories": ["application_logs", "audit_logs", "structured_logs"],
        "description": "Full-text search over OpenSearch log indices",
    },
    {
        "tool_name": "LogPatternAnalysisTool",
        "mcp_server_id": "opensearch",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.LOGS,
        "idempotent": True,
        "expected_latency_seconds": 15,
        "evidence_categories": ["log_patterns", "error_frequency"],
        "description": "Extract recurring log patterns and frequencies",
    },
    {
        "tool_name": "DataDistributionTool",
        "mcp_server_id": "opensearch",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.LOGS,
        "idempotent": True,
        "expected_latency_seconds": 10,
        "evidence_categories": ["field_distribution", "severity_distribution"],
        "description": "Compute field distribution over log data",
    },
    # ── Jaeger / Tempo (Traces) ───────────────────────────────────────────────
    {
        "tool_name": "search_traces",
        "mcp_server_id": "jaeger",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.TRACES,
        "idempotent": True,
        "expected_latency_seconds": 10,
        "evidence_categories": ["distributed_traces", "service_call_graph"],
        "description": "Search for distributed traces by service/operation",
    },
    {
        "tool_name": "get_trace",
        "mcp_server_id": "jaeger",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.TRACES,
        "idempotent": True,
        "expected_latency_seconds": 5,
        "evidence_categories": ["trace_detail", "span_timeline"],
        "description": "Retrieve full trace details by trace ID",
    },
    {
        "tool_name": "find_errors",
        "mcp_server_id": "jaeger",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.TRACES,
        "idempotent": True,
        "expected_latency_seconds": 8,
        "evidence_categories": ["trace_errors", "failed_spans"],
        "description": "Find error spans across traces",
    },
    {
        "tool_name": "search_spans",
        "mcp_server_id": "jaeger",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.TRACES,
        "idempotent": True,
        "expected_latency_seconds": 10,
        "evidence_categories": ["span_patterns", "slow_spans"],
        "description": "Search for spans matching criteria",
    },
    # ── CSV Intelligence Layer (DuckDB — local, in-process) ───────────────────
    # These tools have mcp_server_id='csv_engine'. They are dispatched locally
    # by the CsvSignalAdapter, NOT via a network MCP call.
    {
        "tool_name": "csv_schema_discovery",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 2,
        "evidence_categories": ["available_signals", "data_inventory"],
        "description": "Discover available CSV tables, columns, and row counts. Always call first.",
    },
    {
        "tool_name": "csv_sql_query",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 10,
        "evidence_categories": [
            "metrics_current_state", "error_rate", "metrics_trend",
            "application_logs", "distributed_traces", "field_distribution",
        ],
        "description": "Run SQL on CSV data (200-row cap — use GROUP BY for large datasets).",
    },
    {
        "tool_name": "csv_anomaly_detection",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 5,
        "evidence_categories": ["metrics_anomaly", "error_rate", "latency_p99"],
        "description": "Z-score anomaly detection on prometheus_metrics CSV. Returns only anomalous points.",
    },
    {
        "tool_name": "csv_log_pattern_extract",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 8,
        "evidence_categories": ["log_patterns", "error_logs", "error_frequency"],
        "description": "Deduplicate log lines into patterns with occurrence counts.",
    },
    {
        "tool_name": "csv_slow_span_detection",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 8,
        "evidence_categories": ["slow_spans", "trace_errors", "service_call_graph"],
        "description": "Find abnormally slow trace spans (>2× P95 per service).",
    },
    {
        "tool_name": "csv_time_window_summary",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 5,
        "evidence_categories": ["metrics_trend", "throughput_trend", "log_patterns"],
        "description": "Rolling time-bucket aggregation (min/max/avg/count per window).",
    },
    {
        "tool_name": "csv_correlation_analysis",
        "mcp_server_id": "csv_engine",
        "tier": ToolTier.OBSERVATION,
        "signal_type": SignalType.CSV,
        "idempotent": True,
        "expected_latency_seconds": 6,
        "evidence_categories": ["causal_correlation", "error_rate", "latency_p99"],
        "description": "Pearson correlation between a metric KPI and log error rate across time windows.",
    },
]

# Evidence category → list of tool_names (ranked by expected value)
_CATEGORY_TO_TOOLS: dict[str, list[str]] = {}

def _build_category_index() -> None:
    """Build the evidence_category → tool_name index at module load time."""
    for spec in _TOOL_MANIFEST:
        for cat in spec.get("evidence_categories", []):
            _CATEGORY_TO_TOOLS.setdefault(cat, []).append(spec["tool_name"])

_build_category_index()


# ─── ToolRegistry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Maps evidence gaps to concrete ToolSpecs.

    Phase 1: Uses the hardcoded manifest above.
    Phase 2: Loads from Qdrant C2 collection via ConANN search.
    """

    def __init__(self) -> None:
        self._manifest: list[dict] = _TOOL_MANIFEST
        # Build name → spec index
        self._by_name: dict[str, dict] = {s["tool_name"]: s for s in self._manifest}

    def get_tools_for_gap(self, evidence_category: str) -> list[ToolSpec]:
        """
        Return ToolSpecs for a given evidence gap category.

        Args:
            evidence_category: e.g., 'error_rate', 'application_logs', 'trace_errors'.

        Returns:
            List of ToolSpecs, ranked by expected information value.
            Empty list if no tools cover this category.
        """
        tool_names = _CATEGORY_TO_TOOLS.get(evidence_category, [])
        specs = []
        for name in tool_names:
            raw = self._by_name.get(name)
            if raw:
                specs.append(self._to_tool_spec(raw))
        return specs

    def get_tool_by_name(
        self, tool_name: str, server_id: str
    ) -> Optional[ToolSpec]:
        """Look up a specific tool by name and server_id."""
        raw = self._by_name.get(tool_name)
        if raw and raw["mcp_server_id"] == server_id:
            return self._to_tool_spec(raw)
        return None

    def get_all_tools(self) -> list[ToolSpec]:
        """Return all registered tools."""
        return [self._to_tool_spec(s) for s in self._manifest]

    def get_tools_by_signal_type(self, signal_type: SignalType) -> list[ToolSpec]:
        """Filter tools by signal type."""
        return [
            self._to_tool_spec(s)
            for s in self._manifest
            if s["signal_type"] == signal_type
        ]

    @staticmethod
    def _to_tool_spec(raw: dict) -> ToolSpec:
        return ToolSpec(
            tool_name=raw["tool_name"],
            mcp_server_id=raw["mcp_server_id"],
            tier=raw["tier"],
            signal_type=raw["signal_type"],
            idempotent=raw.get("idempotent", True),
            expected_latency_seconds=raw.get("expected_latency_seconds", 10),
        )


# Module-level singleton
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Return the global ToolRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
