"""
airs_v2.interfaces — External-facing adapters and observability hooks.

Planned modules (Stage 4):
    cli_v2.py         — Typer CLI with streaming Rich output (replaces cli/main.py)
    api_v2.py         — FastAPI router exposing the v2 graph over HTTP/SSE
    otel_tracing.py   — OpenTelemetry span export for every node transition
    langsmith_v2.py   — Enhanced LangSmith trace config with cost & latency metadata
"""
# Stage 0: intentionally empty — scaffold only
