# AIRS — Reliability-First Cognitive Architecture for Incident Response

An autonomous incident response agent built on Temporal, LangGraph, Qdrant, and the Model Context Protocol (MCP).

## Quick Start

```bash
# Start infrastructure
make infra

# Install dependencies
make install

# Ingest knowledge corpus into Qdrant
make ingest

# Start the worker
make workers

# Trigger a test investigation
make trigger ALERT=path/to/alert.json
```

## Architecture

- **Temporal** — Durable workflow orchestration
- **LangGraph** — Deterministic reasoning state machine
- **Qdrant** — 6-collection ConANN vector retrieval
- **Redis** — Entity resolution + idempotency
- **MCP** — Tool execution protocol
for full documentation.
