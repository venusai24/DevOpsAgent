"""
MCP Tool Execution Activity — Module 1.10.

Temporal activity: Execute an MCP tool call via the appropriate SignalAdapter.
Implements the 3-stage pipeline: execute → pre_filter → normalize → EvidenceCandidate.

Temporal contract:
  - Idempotency: checked via Redis before executing. Cache hit → return cached.
  - Retry-safe: MCP client has its own retry logic; activity can be retried safely.
  - Outputs: EvidenceCandidate (pre-filtered, normalized, NOT yet admitted to context)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from temporalio import activity

from airs.mcp.client import MCPToolClient
from airs.mcp.idempotency import IdempotencyManager, create_idempotency_manager
from airs.models.evidence import EvidenceCandidate
from airs.models.intents import ExecutionIntent, SignalType
from airs.signal_adapters import get_adapter

log = logging.getLogger(__name__)


@activity.defn(name="execute_mcp_tool")
async def execute_mcp_tool(
    investigation_id: str,
    hop_index: int,
    intent: ExecutionIntent,
    incident_service: str,
    incident_namespace: str,
) -> EvidenceCandidate:
    """
    Execute an MCP tool and return a pre-filtered EvidenceCandidate.

    Steps:
    1. Check idempotency cache — return cached if hit
    2. Select SignalAdapter for the intent's signal_type
    3. execute(): translate intent → MCP call → RawSignalPayload
    4. pre_filter(): deterministic noise reduction (no LLM)
    5. normalize(): RawSignal → EvidenceCandidate
    6. Cache result in Redis

    Args:
        investigation_id: For idempotency key namespacing.
        hop_index:        Current investigation hop.
        intent:           ExecutionIntent(action=EXECUTE_TOOL) with tool_spec.
        incident_service: Affected service for pre-filter scope.
        incident_namespace: Kubernetes namespace for K8s pre-filters.

    Returns:
        EvidenceCandidate — pre-filtered, normalized, ready for context admission.

    Raises:
        ValueError: If intent has no tool_spec.
    """
    if intent.tool_spec is None:
        raise ValueError("execute_mcp_tool called with intent missing tool_spec")

    spec = intent.tool_spec

    activity.logger.info(
        "Executing MCP tool: %s/%s (hop=%d, idempotent=%s)",
        spec.mcp_server_id, spec.tool_name, hop_index, spec.idempotent,
    )

    # ── Idempotency check ─────────────────────────────────────────────────────
    from airs.config import settings
    idem_manager = create_idempotency_manager(settings.redis_url)
    idem_key = idem_manager.make_key(
        investigation_id=investigation_id,
        hop_index=hop_index,
        tool_name=spec.tool_name,
        arguments=spec.arguments,
    )
    args_hash = idem_manager.arguments_hash(spec.arguments)

    if spec.idempotent:
        cached = idem_manager.get(idem_key)
        if cached is not None:
            activity.logger.info(
                "Idempotency cache hit for %s/%s — returning cached result",
                spec.mcp_server_id, spec.tool_name,
            )
            return EvidenceCandidate(**cached)

    # ── Get signal adapter ────────────────────────────────────────────────────
    adapter = get_adapter(spec.signal_type)

    # ── Execute MCP call ──────────────────────────────────────────────────────
    mcp_client = MCPToolClient.from_settings()
    raw_payload = await adapter.execute(intent, mcp_client)

    # Attach idempotency metadata
    raw_payload.idempotency_key = idem_key
    raw_payload.arguments_hash = args_hash

    # ── Pre-filter ────────────────────────────────────────────────────────────
    filtered = adapter.pre_filter(
        raw_payload,
        incident_service=incident_service,
        incident_namespace=incident_namespace,
    )

    # ── Normalize → EvidenceCandidate ────────────────────────────────────────
    candidate = adapter.normalize(filtered, hop_index=hop_index)

    # ── Cache result ──────────────────────────────────────────────────────────
    if spec.idempotent:
        try:
            idem_manager.set(idem_key, candidate.model_dump(mode="json"))
        except Exception as e:
            log.warning("Idempotency cache write failed: %s", e)

    activity.logger.info(
        "MCP tool executed: %s/%s → entity=%s, tokens=%d",
        spec.mcp_server_id, spec.tool_name,
        candidate.raw_identifiers[:1],
        candidate.token_count,
    )

    return candidate
