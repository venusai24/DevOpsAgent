"""
Logs Signal Adapter (Loki / OpenSearch) — Module 1.6.

Handles LOGS signal type: pattern deduplication pre-filter.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from airs.models.evidence import EvidenceCandidate, EntityType, SignalSource
from airs.models.intents import ExecutionIntent, SignalType
from airs.signal_adapters.base import (
    FilteredSignalPayload,
    RawSignalPayload,
    SignalAdapter,
)
from airs.signal_adapters.prefilters import dedup_log_patterns

log = logging.getLogger(__name__)


class LogsSignalAdapter(SignalAdapter):
    """
    Logs signal adapter for Loki and OpenSearch MCP servers.
    Pre-filter: log pattern deduplication (collapses N repeated lines to 1).
    """

    @property
    def signal_type(self) -> SignalType:
        return SignalType.LOGS

    async def execute(self, intent: ExecutionIntent, mcp_client: Any) -> RawSignalPayload:
        if intent.tool_spec is None:
            raise ValueError("LogsSignalAdapter requires a tool_spec")

        spec = intent.tool_spec
        response = await mcp_client.invoke(
            server_id=spec.mcp_server_id,
            tool_name=spec.tool_name,
            arguments=spec.arguments,
            timeout=float(spec.expected_latency_seconds * 3),
        )
        return RawSignalPayload(
            server_id=spec.mcp_server_id,
            tool_name=spec.tool_name,
            content=response.content,
            raw_text=response.raw_text,
            raw_digest=response.raw_digest,
            idempotency_key="",
            arguments_hash="",
            success=response.success,
            error_code=response.error_code,
            error_message=response.error_message,
        )

    def pre_filter(
        self,
        payload: RawSignalPayload,
        incident_service: Optional[str] = None,
        incident_namespace: Optional[str] = None,
        time_window_minutes: int = 30,
    ) -> FilteredSignalPayload:
        if not payload.success:
            return FilteredSignalPayload(
                server_id=payload.server_id,
                tool_name=payload.tool_name,
                filtered_content={"error": payload.error_message, "success": False},
                token_estimate=20,
                filter_stats={"method": "passthrough", "reason": "tool_failure"},
                raw_digest=payload.raw_digest,
                idempotency_key=payload.idempotency_key,
                arguments_hash=payload.arguments_hash,
                received_at=payload.received_at,
            )

        raw_lines = self._extract_log_lines(payload.content)

        # Service-level pre-filter if service is known
        if incident_service:
            service_lines = [
                l for l in raw_lines if incident_service.lower() in l.lower()
            ]
            raw_lines = service_lines if service_lines else raw_lines

        patterns, stats = dedup_log_patterns(raw_lines, max_patterns=20)

        filtered_content = {
            "service_filter": incident_service,
            "namespace_filter": incident_namespace,
            "log_patterns": patterns,
            "total_raw_lines": len(raw_lines),
        }
        return FilteredSignalPayload(
            server_id=payload.server_id,
            tool_name=payload.tool_name,
            filtered_content=filtered_content,
            token_estimate=self._estimate_tokens(filtered_content),
            filter_stats=stats,
            raw_digest=payload.raw_digest,
            idempotency_key=payload.idempotency_key,
            arguments_hash=payload.arguments_hash,
            received_at=payload.received_at,
        )

    def normalize(self, filtered: FilteredSignalPayload, hop_index: int) -> EvidenceCandidate:
        content = filtered.filtered_content
        service = content.get("service_filter", "unknown-service")
        identifiers = [service] if service else ["unknown-service"]

        return EvidenceCandidate(
            entity_type=EntityType.SERVICE,
            raw_identifiers=identifiers,
            signal_source=SignalSource.LOGS,
            filtered_content=content,
            mcp_server_id=filtered.server_id,
            tool_invoked=filtered.tool_name,
            query_parameters_hash=filtered.arguments_hash,
            raw_response_digest=filtered.raw_digest,
            idempotency_key=filtered.idempotency_key,
            observed_at=filtered.received_at,
            valid_from=filtered.received_at,
            valid_until=None,
            hop_index=hop_index,
            token_count=filtered.token_estimate,
        )

    @staticmethod
    def _extract_log_lines(content: Any) -> list[str]:
        """Extract flat log line strings from various Loki/OpenSearch response formats."""
        if isinstance(content, list):
            lines = []
            for item in content:
                if isinstance(item, str):
                    lines.append(item)
                elif isinstance(item, dict):
                    # Loki: {"stream": {...}, "values": [[ts, line], ...]}
                    for ts, line in item.get("values", []):
                        if isinstance(line, str):
                            lines.append(line)
                    # OpenSearch: {"_source": {"message": "..."}}
                    msg = item.get("_source", {}).get("message", "")
                    if msg:
                        lines.append(str(msg))
            return lines
        if isinstance(content, dict):
            # Loki stream format
            results = content.get("data", {}).get("result", [])
            lines = []
            for stream in results:
                for _, line in stream.get("values", []):
                    lines.append(str(line))
            return lines
        return []
