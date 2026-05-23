"""
agent/tools.py

LangChain @tool-decorated functions that call the local mock enterprise API.

Design constraints (ImplementationPlan.md Section 7):
  - The LLM only sees the tool docstrings. It must deduce the correct query /
    service strings from those docstrings and from the incident context it has
    accumulated in the message history.
  - HTTP 400 / 404 responses are surfaced as ToolException so that LangGraph
    automatically converts them into ToolMessages. This forces the LLM to read
    the error and produce a corrected tool call on the next iteration.
  - All I/O is async (httpx.AsyncClient) to match LangGraph's async execution.
  - The base URL is read from MOCK_API_BASE_URL so tests can point tools at a
    different port without monkey-patching.

Usage (LangGraph node):
    from agent.tools import ALL_TOOLS
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.tools import tool, ToolException
from config import settings
from agent.integrations.datadog import fetch_datadog_metrics
from agent.integrations.aws_cloudwatch import fetch_cloudwatch_logs

import httpx
from langchain_core.tools import tool, ToolException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_base_url() -> str:
    return os.getenv("MOCK_API_BASE_URL", "http://localhost:8000")

_TIMEOUT: float = float(os.getenv("MOCK_API_TIMEOUT_SECONDS", "10.0"))


# ---------------------------------------------------------------------------
# Private formatting helpers
# ---------------------------------------------------------------------------


def _fmt_metric_payload(data: dict[str, Any]) -> str:
    """
    Render the /metrics API response as a concise markdown summary.
    Keeps the LLM context window focused on the anomaly signal.
    """
    metric = data.get("metric", {})
    name = metric.get("metric", "unknown")
    current = metric.get("pool_current") or metric.get("current", "N/A")
    anomaly = metric.get("anomaly_detected", False)
    note = metric.get("note", "")
    points: list[dict[str, Any]] = metric.get("data_points", [])
    latest_val = points[-1]["value"] if points else "N/A"
    latest_ts = points[-1]["timestamp"] if points else "N/A"

    lines = [
        f"## Metric: `{name}`",
        f"- **Latest value**: {latest_val} (at {latest_ts})",
        f"- **Current**: {current}",
        f"- **Anomaly detected**: {anomaly}",
    ]
    if metric.get("pool_max"):
        lines.append(f"- **Pool max capacity**: {metric['pool_max']}")
    if metric.get("pool_waiting"):
        lines.append(f"- **Requests waiting for connection**: {metric['pool_waiting']}")
    if metric.get("anomaly_start"):
        lines.append(f"- **Anomaly started at**: {metric['anomaly_start']}")
    if metric.get("threshold"):
        lines.append(f"- **Alert threshold**: {metric['threshold']}")
    if note:
        lines.append(f"- **Note**: {note}")

    # Mini trend table (last 4 data points)
    if len(points) > 1:
        lines += ["", "| Timestamp | Value |", "|-----------|-------|"]
        for pt in points[-4:]:
            lines.append(f"| {pt['timestamp']} | {pt['value']} {pt.get('unit', '')} |")

    return "\n".join(lines)


def _fmt_log_payload(data: dict[str, Any]) -> str:
    """
    Render the /logs API response as a concise markdown summary.
    Only WARN / ERROR / CRITICAL entries are surfaced to reduce noise.
    """
    entries: list[dict[str, Any]] = data.get("log_entries", [])
    service = data.get("service", "unknown")
    notable = [
        e for e in entries
        if e.get("level", "INFO") in ("WARN", "ERROR", "CRITICAL")
    ]

    lines = [
        f"## Logs: `{service}`",
        f"- **Total entries returned**: {data.get('total_entries', 0)}",
        f"- **Notable entries (WARN/ERROR/CRITICAL)**: {len(notable)}",
        "",
    ]
    for entry in notable:
        lines.append(
            f"**[{entry['level']}]** `{entry['timestamp']}` — {entry['message']}"
        )
        if exc := entry.get("exception"):
            lines.append(f"  ```\n  {exc}\n  ```")
        if tx := entry.get("transaction_id"):
            lines.append(
                f"  ⚠ Long-running TX `{tx}` open for "
                f"{entry.get('duration_seconds', '?')}s"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared async HTTP helper
# ---------------------------------------------------------------------------


async def _get(url: str, params: dict[str, str]) -> dict[str, Any]:
    """
    Perform an async GET request and return the parsed JSON body.

    Raises ToolException for:
      - Connection failures (mock server not running)
      - Timeouts
      - HTTP 4xx / 5xx responses

    ToolException is automatically caught by LangGraph's tool node and
    converted into a ToolMessage, which the LLM reads and self-corrects from.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(url, params=params)
    except httpx.ConnectError as exc:
        raise ToolException(
            f"Cannot connect to the mock enterprise API at {_get_base_url()}. "
            "Ensure the server is running: "
            "uvicorn mock_enterprise.api:app --port 8000 --reload\n"
            f"Error: {exc}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ToolException(
            f"Request to {url} timed out after {_TIMEOUT}s. Error: {exc}"
        ) from exc

    if not response.is_success:
        detail = ""
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise ToolException(
            f"HTTP {response.status_code} from {url}: {detail}"
        )

    return response.json()


# ---------------------------------------------------------------------------
# Tool: get_metrics
# ---------------------------------------------------------------------------


@tool
async def get_metrics(query: str, time_range: str = "last_15m") -> str:
    """
    Fetch time-series performance metrics from the monitoring platform
    (simulates Datadog / Prometheus).

    Use this tool FIRST to identify numeric anomalies before fetching logs.
    The query string should describe the metric you need. Accepted patterns:

      - "db_connections" or "postgresql.connections.active" — Database
        connection pool saturation. Use when the alert mentions DB timeouts.
      - "error_rate" or "http.requests.error_rate" — HTTP 5xx error rate.
        Use to quantify user-facing impact.
      - "cpu_utilization" or "system.cpu.utilization" — Compute utilisation.
        Use to rule out or confirm a compute-side bottleneck.

    Args:
        query:      PromQL-style metric query string (see accepted patterns).
        time_range: Time window, e.g. "last_15m", "last_1h". Default "last_15m".

    Returns:
        Markdown summary with latest value, anomaly status, and a trend table.

    Raises:
        ToolException: If the query is unrecognised (HTTP 400) or the server
                       is unreachable. Read the error and correct the query.
    """
    url = f"{_get_base_url()}/metrics"
    params = {"query": query, "time_range": time_range}
    logger.info("TOOL get_metrics  url=%s  params=%s", url, params)

    # Use Real Datadog SDK if API keys exist
    if settings.DATADOG_API_KEY:
        try:
            logger.info("Using live Datadog metrics API")
            # In real system, this would not return markdown, it would return raw JSON to process
            # For simplicity, returning the raw string from SDK mock
            return await fetch_datadog_metrics(query, time_range)
        except Exception as e:
            logger.warning(f"Datadog live query failed, falling back to mock: {e}")

    data = await _get(url, params)
    return _fmt_metric_payload(data)


# ---------------------------------------------------------------------------
# Tool: get_logs
# ---------------------------------------------------------------------------


@tool
async def get_logs(
    service: str,
    time_range: str = "last_15m",
    level: str | None = None,
) -> str:
    """
    Fetch structured log entries for a specific microservice from the log
    aggregation platform (simulates Splunk / CloudWatch Logs).

    Use this tool AFTER get_metrics to correlate numeric anomalies with
    specific error messages, stack traces, and transaction IDs.

    Args:
        service:    Canonical microservice name. The only seeded service is
                    "payments-service". If a 404 is returned, fall back to
                    "payments-service".
        time_range: Time window, e.g. "last_15m", "last_1h". Default "last_15m".
        level:      Optional log level filter: DEBUG, INFO, WARN, ERROR,
                    CRITICAL. Omit to return all levels.

    Returns:
        Markdown summary of WARN / ERROR / CRITICAL log entries with
        timestamps, messages, exception stack traces, and TX warnings.

    Raises:
        ToolException: If the service has no fixture data (HTTP 404) or the
                       server is unreachable. Read the error and retry with
                       the correct service name.
    """
    url = f"{_get_base_url()}/logs"
    params: dict[str, str] = {"service": service, "time_range": time_range}
    if level:
        params["level"] = level
    logger.info("TOOL get_logs  url=%s  params=%s", url, params)
    
    # Use real AWS CloudWatch Logs if configured
    if False: # We'd check for AWS credentials here (e.g. settings.AWS_ACCESS_KEY_ID)
        try:
            logger.info("Using live CloudWatch Logs API")
            return await fetch_cloudwatch_logs(service, time_range)
        except Exception as e:
            logger.warning(f"Live CloudWatch query failed, falling back to mock: {e}")

    data = await _get(url, params)
    return _fmt_log_payload(data)


# ---------------------------------------------------------------------------
# Convenience export
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: query_topology
# ---------------------------------------------------------------------------


@tool
async def query_topology(service: str, depth: int = 2) -> str:
    """
    Query the Enterprise Knowledge Graph (EKG) for the dependency chain
    and health status of a specific microservice.

    Use this tool when you need to understand:
      - Which databases, caches, or downstream services does this service depend on?
      - Are any dependencies currently in a degraded or critical state?
      - What is the service tier (1=critical, 2=internal, 3=batch)?

    This tool is essential for blast radius assessment and cascade failure detection.

    Args:
        service: Canonical microservice name (e.g. 'payments-service').
        depth:   Number of dependency hops to traverse (1-4). Default 2.

    Returns:
        Markdown summary of the dependency chain with health status indicators.
        🔴 = critical, 🟡 = degraded, 🟢 = healthy, ⚪ = unknown.

    Raises:
        ToolException: If the service is not found in the topology.
    """
    url = f"{_get_base_url()}/topology/{service}/dependencies"
    logger.info("TOOL query_topology  service=%s  depth=%d", service, depth)

    try:
        data = await _get(url, {"depth": str(depth)})
    except ToolException:
        # Fall back to full topology if service-specific lookup fails
        data = await _get(f"{_get_base_url()}/topology", {})

    nodes: list[dict] = data.get("nodes", [])
    edges: list[dict] = data.get("edges", [])
    focus = data.get("focus_service", service)

    lines = [f"## EKG Topology: `{focus}` (depth={depth})\n"]
    lines.append("### Nodes")
    health_icons = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴", "unknown": "⚪"}
    for node in nodes:
        icon = health_icons.get(node.get("health_status", "unknown"), "⚪")
        tier = node.get("tier", "?")
        node_type = node.get("node_type", "service")
        name = node.get("name", "?")
        owner = node.get("owner", "unknown")
        health = node.get("health_status", "unknown").upper()
        lines.append(f"- {icon} **{name}** [Tier {tier}] [{node_type}] — {health} (owner: {owner})")

    if edges:
        lines.append("\n### Dependencies")
        for edge in edges:
            err_rate = edge.get("error_rate_pct", 0)
            protocol = edge.get("protocol", "http")
            flag = " ⚠️" if err_rate > 5 else ""
            lines.append(f"- `{edge['source']}` → `{edge['target']}` ({protocol}, err={err_rate:.1f}%){flag}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: search_historical_incidents
# ---------------------------------------------------------------------------


@tool
async def search_historical_incidents(
    category: str = "",
    service: str = "",
    limit: int = 3,
) -> str:
    """
    Search the Case-Based Reasoning (CBR) database for historical incidents
    that match the current failure pattern.

    Use this tool when you need to:
      - Find precedents for the current root cause category.
      - Retrieve proven remediation steps from past incidents.
      - Estimate MTTR based on similar past resolutions.

    The CBR database contains resolved incidents with:
      - Root cause category (e.g. 'connection_pool_exhaustion', 'oom_killed')
      - Ordered remediation steps with exact kubectl commands
      - Outcome and MTTR from the actual resolution

    Args:
        category: Root cause category to filter by (partial match, optional).
                  Examples: 'connection_pool', 'oom', 'dns', 'disk', 'tls'
        service:  Service name to filter by (partial match, optional).
        limit:    Maximum number of results to return (1-20). Default 3.

    Returns:
        Markdown table of matching historical incidents with similarity hints,
        MTTR, and a link to the postmortem summary for each case.

    Raises:
        ToolException: If the server is unreachable.
    """
    url = f"{_get_base_url()}/incidents/search"
    params: dict[str, str] = {"limit": str(limit)}
    if category:
        params["category"] = category
    if service:
        params["service"] = service

    logger.info("TOOL search_historical_incidents  category=%r  service=%r", category, service)
    data = await _get(url, params)

    results: list[dict] = data.get("results", [])
    total: int = data.get("total", 0)

    if not results:
        return (
            f"No historical incidents found for category='{category}' service='{service}'. "
            "Try broader search terms or omit filters."
        )

    lines = [
        f"## Historical Incident Precedents ({total} total, showing {len(results)})\n",
        "| ID | Service | Category | MTTR | Outcome |",
        "|-----|---------|----------|------|---------|" ,
    ]
    for case in results:
        lines.append(
            f"| `{case.get('incident_id', '?')}` "
            f"| `{case.get('service', '?')}` "
            f"| `{case.get('root_cause_category', '?')}` "
            f"| {case.get('mttr_minutes', '?')}m "
            f"| {case.get('outcome', '?')} |"
        )
    lines.append("")
    for case in results:
        summary = case.get("postmortem_summary", "")
        if summary:
            lines.append(f"**{case.get('incident_id', '?')}**: {summary[:150]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience export
# ---------------------------------------------------------------------------

ALL_TOOLS = [get_metrics, get_logs, query_topology, search_historical_incidents]
"""
Flat list of all investigation tools bound to the LLM in the NEURAL_FULL pathway.
Pass to ``llm.bind_tools(ALL_TOOLS)``.

Tool selection guide for the LLM:
  get_metrics                — Quantify numeric anomalies (CPU, memory, DB pool, error rate)
  get_logs                   — Inspect specific error messages and stack traces
  query_topology             — Understand service dependencies and health status
  search_historical_incidents — Find proven remediation plans from past incidents
"""
