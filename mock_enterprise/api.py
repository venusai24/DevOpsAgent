"""
mock_enterprise/api.py

Simulates Datadog (metrics) and Splunk (logs) monitoring APIs for the
Autonomous Incident Response System (AIRS) demo environment.

The payloads are seeded for a specific incident scenario:
    - payments-service database connection pool exhaustion (P0 severity)

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

# Load fixtures once at startup so every request is sub-millisecond.
_FIXTURES_PATH = Path(__file__).parent / "fixtures.json"
with _FIXTURES_PATH.open() as _f:
    _FIXTURES: dict[str, Any] = json.load(_f)

_INCIDENT_KEY = "db_connection_exhaustion"
_INCIDENT = _FIXTURES["incidents"][_INCIDENT_KEY]

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


def _resolve_metric(query: str) -> dict[str, Any] | None:
    """Map a free-form PromQL-style query string to a fixture payload."""
    q = query.lower().strip()
    if any(kw in q for kw in ("db_conn", "connection", "postgresql")):
        return _INCIDENT["metrics"]["db_connections"]
    if any(kw in q for kw in ("error", "http")):
        return _INCIDENT["metrics"]["error_rate"]
    if any(kw in q for kw in ("cpu", "compute", "system")):
        return _INCIDENT["metrics"]["cpu_utilization"]
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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

    payload = _resolve_metric(query)
    if payload is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unrecognised metric query: {query!r}. "
                f"Accepted patterns: {sorted(_VALID_METRIC_QUERIES)}."
            ),
        )

    response_body = {
        "status": "success",
        "incident_id": _INCIDENT["alert"]["id"],
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

    # Only payments-service is seeded; return 404 for everything else so the
    # LLM is forced to correct its parameter and retry.
    if service.lower() not in ("payments-service", "payments_service"):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No log fixtures found for service: {service!r}. "
                "Available services: ['payments-service']."
            ),
        )

    logs: list[dict[str, Any]] = _INCIDENT["logs"]

    # Optional server-side level filter.
    if level:
        logs = [e for e in logs if e.get("level", "").upper() == level.upper()]

    response_body = {
        "status": "success",
        "incident_id": _INCIDENT["alert"]["id"],
        "service": service,
        "time_range": time_range,
        "total_entries": len(logs),
        "log_entries": logs,
    }
    return JSONResponse(content=response_body)


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
    return JSONResponse(content=_INCIDENT["alert"])
