"""
mock_enterprise/api.py

Simulates the full AIRS enterprise telemetry and knowledge layer:

  Telemetry endpoints:
    GET /metrics          — Datadog / Prometheus metric time-series
    GET /logs             — Splunk / CloudWatch log entries

  Enterprise Knowledge Graph (EKG) endpoints:
    GET /topology                           — Full infrastructure topology
    GET /topology/{service}/dependencies    — Dependency chain for a service
    POST /topology/blast-radius             — Blast radius calculation
    PATCH /topology/{service}/health        — Update service health status

  Case-Based Reasoning (CBR) endpoints:
    GET /incidents/search                   — Cosine-similarity incident search
    POST /incidents                         — Store a resolved incident case

  Meta endpoints:
    GET /health                             — Liveness probe
    GET /alert                              — Active incident alert payload
    POST /active_incident                   — Switch the active incident scenario

Run with:
    uvicorn mock_enterprise.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_enterprise.api")

app = FastAPI(
    title="AIRS Mock Enterprise API",
    description=(
        "Simulates Datadog (metrics) and Splunk (logs) endpoints for the "
        "Autonomous Incident Response System demo environment."
    ),
    version="1.0.0",
)

# Load incident fixtures once at startup.
_FIXTURES_PATH = Path(__file__).parent / "fixtures.json"
with _FIXTURES_PATH.open() as _f:
    _FIXTURES: dict[str, Any] = json.load(_f)

# Load topology fixtures (EKG) once at startup.
_TOPOLOGY_PATH = Path(__file__).parent / "topology_fixtures.json"
_TOPOLOGY: dict[str, Any] = {}
if _TOPOLOGY_PATH.exists():
    with _TOPOLOGY_PATH.open() as _tf:
        _TOPOLOGY = json.load(_tf)
else:
    logger.warning("topology_fixtures.json not found — EKG endpoints will return empty data.")

_active_incident_key = "db_connection_exhaustion"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_METRIC_QUERIES = {
    "db_connections",
    "postgresql.connections.active",
    "error_rate",
    "http.requests.error_rate",
    "cpu",
    "cpu_utilization",
    "system.cpu.utilization",
}


def _resolve_metric(query: str):
    """Map a free-form PromQL-style query string to a fixture payload."""
    q = query.lower().strip()
    
    inc = _FIXTURES["incidents"].get(_active_incident_key)
    if not inc:
        return None
        
    metrics = inc.get("metrics", {})
    # Direct string matching against seeded metrics
    for m_key, m_payload in metrics.items():
        if m_key in q or m_payload["metric"] in q:
            return m_payload, inc["alert"]["id"]
    
    # Legacy keyword mapping for db_connection_exhaustion scenario
    if inc["alert"]["service"] == "payments-service":
        if any(kw in q for kw in ("db_conn", "connection", "postgresql")):
            return metrics.get("db_connections"), inc["alert"]["id"]
        if any(kw in q for kw in ("error", "http")):
            return metrics.get("error_rate"), inc["alert"]["id"]
        if any(kw in q for kw in ("cpu", "compute", "system")):
            return metrics.get("cpu_utilization"), inc["alert"]["id"]
            
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/active_incident", tags=["Meta"])
async def set_active_incident(payload: dict) -> dict:
    global _active_incident_key
    _active_incident_key = payload.get("incident_key", "db_connection_exhaustion")
    logger.info("Set active incident key to: %s", _active_incident_key)
    return {"status": "ok", "active_incident": _active_incident_key}


@app.get(
    "/metrics",
    summary="Query Mock Monitoring Metrics (Datadog / Prometheus)",
    response_description="Time-series metric payload for the requested query.",
    tags=["Telemetry"],
)
async def get_metrics(
    query: str = Query(
        ...,
        description=(
            "PromQL-style metric query string. "
            "Recognised patterns: 'db_connections', 'error_rate', 'cpu_utilization'. "
            "Returns 400 if query is empty or unrecognised."
        ),
        min_length=1,
        examples={
            "db_connections": {
                "summary": "Database connection pool utilisation",
                "value": "db_connections",
            },
            "error_rate": {
                "summary": "HTTP error rate",
                "value": "error_rate",
            },
            "cpu": {
                "summary": "CPU utilisation",
                "value": "cpu_utilization",
            },
        },
    ),
    time_range: str = Query(
        "last_15m",
        description="Time window for the query (informational only in this mock).",
    ),
) -> JSONResponse:
    """
    Return simulated time-series metrics for the active incident scenario.

    **Incident context**: payments-service database connection pool exhaustion.

    The endpoint accepts a free-form `query` string and maps it to one of three
    pre-seeded metric payloads:

    | Query pattern         | Returned metric                        |
    |-----------------------|----------------------------------------|
    | `db_connections` / `connection` / `postgresql` | DB pool saturation |
    | `error_rate` / `http` | HTTP 5xx error rate spike              |
    | `cpu` / `compute`     | CPU utilisation (nominal — no anomaly) |

    Returns **400** if the query string is empty or cannot be mapped to a known
    metric, forcing the calling LLM to self-correct its tool argument.
    """
    logger.info("GET /metrics  query=%r  time_range=%r", query, time_range)

    resolved = _resolve_metric(query)
    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unrecognised metric query: {query!r}. "
                f"Accepted patterns: {sorted(_VALID_METRIC_QUERIES)}."
            ),
        )

    payload, incident_id = resolved
    
    response_body = {
        "status": "success",
        "incident_id": incident_id,
        "query": query,
        "time_range": time_range,
        "metric": payload,
    }
    return JSONResponse(content=response_body)


@app.get(
    "/logs",
    summary="Query Mock Log Aggregation (Splunk / CloudWatch)",
    response_description="Chronological array of structured log entries for the requested service.",
    tags=["Telemetry"],
)
async def get_logs(
    service: str = Query(
        ...,
        description=(
            "Name of the microservice to retrieve logs for. "
            "Only 'payments-service' is seeded with incident data."
        ),
        min_length=1,
        examples={
            "payments-service": {
                "summary": "Fetch logs for the payments microservice",
                "value": "payments-service",
            },
        },
    ),
    time_range: str = Query(
        "last_15m",
        description="Time window for log retrieval (informational only in this mock).",
    ),
    level: str | None = Query(
        None,
        description="Filter by log level (DEBUG, INFO, WARN, ERROR, CRITICAL). Optional.",
    ),
) -> JSONResponse:
    """
    Return simulated structured log entries for the specified service.

    **Incident context**: payments-service database connection pool exhaustion.

    The log stream contains `WARN`, `ERROR`, and `CRITICAL` entries that include:

    - SQLAlchemy `TimeoutError` stack traces confirming pool exhaustion
    - Long-running transaction IDs indicating a potential connection leak
    - Downstream HTTP 503 cascades to the payment gateway

    Returns **404** if the requested `service` has no fixtures, forcing the
    calling LLM to use the correct service name.

    Optionally filter by `level` (case-insensitive).
    """
    logger.info(
        "GET /logs  service=%r  time_range=%r  level=%r", service, time_range, level
    )

    logs = []
    incident_id = "unknown"
    inc = _FIXTURES["incidents"].get(_active_incident_key)
    if inc and inc["alert"]["service"].lower() == service.lower().replace("_", "-"):
        logs = inc["logs"]
        incident_id = inc["alert"]["id"]
    else:
        for i_val in _FIXTURES["incidents"].values():
            if i_val["alert"]["service"].lower() == service.lower().replace("_", "-"):
                logs = i_val["logs"]
                incident_id = i_val["alert"]["id"]
                break

    if not logs:
        available = [i_val["alert"]["service"] for i_val in _FIXTURES["incidents"].values()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"No log fixtures found for service: {service!r}. "
                f"Available services: {available}."
            ),
        )

    # Optional server-side level filter.
    if level:
        logs = [e for e in logs if e.get("level", "").upper() == level.upper()]

    response_body = {
        "status": "success",
        "incident_id": incident_id,
        "service": service,
        "time_range": time_range,
        "total_entries": len(logs),
        "log_entries": logs,
    }
    return JSONResponse(content=response_body)


# ---------------------------------------------------------------------------
# EKG Topology endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/topology",
    summary="Full infrastructure topology",
    tags=["EKG"],
)
async def get_topology() -> JSONResponse:
    """
    Return the full enterprise topology graph: all services, databases,
    caches, external dependencies, and their directed dependency edges.
    """
    logger.info("GET /topology")
    return JSONResponse(content={
        "status": "ok",
        "nodes": _TOPOLOGY.get("nodes", []),
        "edges": _TOPOLOGY.get("edges", []),
        "node_count": len(_TOPOLOGY.get("nodes", [])),
        "edge_count": len(_TOPOLOGY.get("edges", [])),
    })


@app.get(
    "/topology/{service}/dependencies",
    summary="Dependency chain for a specific service",
    tags=["EKG"],
)
async def get_service_dependencies(
    service: str,
    depth: int = Query(2, ge=1, le=4, description="Traversal depth (1-4 hops)."),
) -> JSONResponse:
    """
    Return all services that ``service`` directly or transitively depends on,
    up to ``depth`` hops. Also returns the health status of each dependency.
    """
    logger.info("GET /topology/%s/dependencies  depth=%d", service, depth)

    nodes = _TOPOLOGY.get("nodes", [])
    edges = _TOPOLOGY.get("edges", [])

    # Find the focal node
    focal = next((n for n in nodes if n["name"].lower() == service.lower()), None)
    if focal is None:
        raise HTTPException(
            status_code=404,
            detail=f"Service '{service}' not found in topology. "
                   f"Available: {[n['name'] for n in nodes[:10]]}"
        )

    # BFS traversal up to `depth` hops
    visited: set[str] = {service}
    frontier: set[str] = {service}
    dep_nodes: list[dict] = [focal]
    dep_edges: list[dict] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for edge in edges:
            if edge.get("source") in frontier and edge.get("target") not in visited:
                target_name = edge["target"]
                next_frontier.add(target_name)
                visited.add(target_name)
                dep_edges.append(edge)
                target_node = next((n for n in nodes if n["name"] == target_name), None)
                if target_node:
                    dep_nodes.append(target_node)
        frontier = next_frontier
        if not frontier:
            break

    return JSONResponse(content={
        "status": "ok",
        "focus_service": service,
        "depth": depth,
        "nodes": dep_nodes,
        "edges": dep_edges,
        "dependency_count": len(dep_nodes) - 1,
    })


@app.post(
    "/topology/blast-radius",
    summary="Calculate blast radius for a target service",
    tags=["EKG"],
)
async def calculate_blast_radius(payload: dict) -> JSONResponse:
    """
    Calculate the blast radius (upstream dependents) of a target service.

    Unlike dependency traversal (downstream), blast radius finds all services
    that **depend on** the target and would be impacted if it fails.

    Body:
        {"service": "payments-service"}
    """
    service = payload.get("service", "")
    logger.info("POST /topology/blast-radius  service=%s", service)

    nodes = _TOPOLOGY.get("nodes", [])
    edges = _TOPOLOGY.get("edges", [])

    # Find the focal node
    focal = next((n for n in nodes if n["name"].lower() == service.lower()), None)
    if focal is None:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found.")

    # BFS upstream: find services that depend ON this service
    visited: set[str] = {service}
    frontier: set[str] = {service}
    affected: list[str] = []
    tier1_services: list[str] = []
    on_call_contacts: list[str] = []

    for _ in range(4):  # Max 4 hops upstream
        next_frontier: set[str] = set()
        for edge in edges:
            # Reverse direction: find who depends on services in frontier
            if edge.get("target") in frontier and edge.get("source") not in visited:
                src = edge["source"]
                next_frontier.add(src)
                visited.add(src)
                affected.append(src)
                src_node = next((n for n in nodes if n["name"] == src), None)
                if src_node:
                    if src_node.get("tier", 3) == 1:
                        tier1_services.append(src)
                    if src_node.get("on_call"):
                        on_call_contacts.append(src_node["on_call"])
        frontier = next_frontier
        if not frontier:
            break

    tier1_impact = bool(tier1_services)
    risk_score = min(1.0, round(
        (len(affected) / max(1, len(nodes))) * 0.5 +
        (1.0 if tier1_impact else 0.0) * 0.5,
        2
    ))
    recommendation = "block" if risk_score >= 0.8 else (
        "require_approval" if tier1_impact or risk_score >= 0.3 else "auto_execute"
    )

    return JSONResponse(content={
        "status": "ok",
        "target_service": service,
        "affected_services": affected,
        "tier1_services": tier1_services,
        "tier1_impact": tier1_impact,
        "risk_score": risk_score,
        "recommendation": recommendation,
        "on_call_contacts": list(set(on_call_contacts)),
        "estimated_user_impact_pct": round(risk_score * 100, 1),
    })


@app.patch(
    "/topology/{service}/health",
    summary="Update service health status",
    tags=["EKG"],
)
async def update_service_health(service: str, payload: dict) -> JSONResponse:
    """
    Update the health_status of a node in the topology fixture.
    Used by topology_agent_node to mark a service as 'critical' during an incident.

    Body: {"health_status": "critical" | "degraded" | "healthy" | "unknown"}
    """
    status = payload.get("health_status", "unknown")
    logger.info("PATCH /topology/%s/health  status=%s", service, status)

    for node in _TOPOLOGY.get("nodes", []):
        if node["name"].lower() == service.lower():
            node["health_status"] = status
            return JSONResponse(content={"status": "ok", "service": service, "health_status": status})

    raise HTTPException(status_code=404, detail=f"Service '{service}' not found.")


# ---------------------------------------------------------------------------
# CBR Incident endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/incidents/search",
    summary="Search historical incidents by category",
    tags=["CBR"],
)
async def search_incidents(
    category: str = Query("", description="Root cause category filter (partial match)."),
    service: str = Query("", description="Service name filter."),
    limit: int = Query(5, ge=1, le=20, description="Maximum results to return."),
) -> JSONResponse:
    """
    Search the seeded historical incident database for cases matching the
    given category or service name. Used by the CBR diagnostic agent to
    surface relevant precedents during investigation.
    """
    logger.info(
        "GET /incidents/search  category=%r  service=%r  limit=%d",
        category, service, limit,
    )

    # Source results from topology_fixtures.json historical_incidents section
    all_cases = _TOPOLOGY.get("historical_incidents", [])

    # Also fold in incident fixtures
    for inc in _FIXTURES.get("incidents", {}).values():
        alert = inc.get("alert", {})
        all_cases.append({
            "incident_id": alert.get("id", "unknown"),
            "service": alert.get("service", "unknown"),
            "root_cause_category": inc.get("root_cause_category", "unknown"),
            "severity": alert.get("severity", "P1"),
            "outcome": "resolved",
            "mttr_minutes": inc.get("mttr_minutes", 15),
            "postmortem_summary": alert.get("description", "")[:200],
        })

    # Filter
    results = [
        c for c in all_cases
        if (not category or category.lower() in c.get("root_cause_category", "").lower())
        and (not service or service.lower() in c.get("service", "").lower())
    ]

    return JSONResponse(content={
        "status": "ok",
        "total": len(results),
        "results": results[:limit],
    })


@app.post(
    "/incidents",
    summary="Store a resolved incident case",
    tags=["CBR"],
    status_code=201,
)
async def store_incident(payload: dict) -> JSONResponse:
    """
    Store a newly resolved incident as a historical case.
    Called by retain_node after successful remediation to close
    the continuous learning loop.
    """
    incident_id = payload.get("incident_id", "unknown")
    logger.info("POST /incidents  incident_id=%s", incident_id)

    # Append to in-memory topology historical_incidents list
    if "historical_incidents" not in _TOPOLOGY:
        _TOPOLOGY["historical_incidents"] = []
    _TOPOLOGY["historical_incidents"].append(payload)

    return JSONResponse(
        status_code=201,
        content={"status": "stored", "incident_id": incident_id},
    )


# ---------------------------------------------------------------------------
# Health / Meta
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Meta"], summary="Health check")
async def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok", "environment": "mock_enterprise"}


@app.get("/alert", tags=["Meta"], summary="Active incident alert payload")
async def get_alert() -> JSONResponse:
    """
    Returns the seeded PagerDuty-style alert that triggered the current
    incident scenario. Useful for bootstrapping the agent's initial context.
    """
    incident = list(_FIXTURES["incidents"].values())[0]
    return JSONResponse(content=incident["alert"])
