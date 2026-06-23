"""
Traces Signal Adapter (Jaeger/Tempo) — Module 1.6.

Pre-filter: error span extraction. Retains only error spans or
slowest spans from the raw Jaeger/Tempo response.
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
from airs.signal_adapters.prefilters import filter_error_spans

log = logging.getLogger(__name__)


class TracesSignalAdapter(SignalAdapter):
    """Distributed traces signal adapter for Jaeger/Tempo MCP."""

    @property
    def signal_type(self) -> SignalType:
        return SignalType.TRACES

    async def execute(self, intent: ExecutionIntent, mcp_client: Any) -> RawSignalPayload:
        if intent.tool_spec is None:
            raise ValueError("TracesSignalAdapter requires a tool_spec")

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

        spans = self._extract_spans(payload.content)
        filtered_spans, stats = filter_error_spans(spans, max_spans=20)

        # Build service-level summary
        services_involved = list({
            s.get("process", {}).get("serviceName", "unknown")
            for s in spans
            if isinstance(s.get("process"), dict)
        })

        filtered_content = {
            "service_filter": incident_service,
            "services_involved": services_involved[:10],
            "error_spans": filtered_spans,
            "total_spans": len(spans),
            "trace_count": self._count_unique_traces(spans),
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
            signal_source=SignalSource.TRACES,
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
    def _extract_spans(content: Any) -> list[dict]:
        """Extract span list from Jaeger/Tempo response."""
        if isinstance(content, list):
            return [s for s in content if isinstance(s, dict)]
        if isinstance(content, dict):
            # Jaeger: {data: [{spans: [...]}]}
            data = content.get("data", [])
            if isinstance(data, list):
                spans = []
                for trace in data:
                    if isinstance(trace, dict):
                        spans.extend(trace.get("spans", []))
                return spans
        return []

    @staticmethod
    def _count_unique_traces(spans: list[dict]) -> int:
        return len({s.get("traceID") for s in spans if "traceID" in s})


class K8sStateSignalAdapter(SignalAdapter):
    """
    Kubernetes state signal adapter (kubectl MCP).
    Pre-filter: K8s event filtering (Warning events only).
    """

    @property
    def signal_type(self) -> SignalType:
        return SignalType.K8S_STATE

    async def execute(self, intent: ExecutionIntent, mcp_client: Any) -> RawSignalPayload:
        if intent.tool_spec is None:
            raise ValueError("K8sStateSignalAdapter requires a tool_spec")

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
        from airs.signal_adapters.prefilters import filter_k8s_events

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

        events = self._extract_events(payload.content)
        filtered_events, stats = filter_k8s_events(
            events,
            namespace=incident_namespace,
            max_events=30,
        )

        filtered_content = {
            "namespace_filter": incident_namespace,
            "service_filter": incident_service,
            "events": filtered_events,
            "total_events": len(events),
            "raw_k8s_data": self._extract_non_events(payload.content),
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
        namespace = content.get("namespace_filter", "unknown")
        service = content.get("service_filter", "unknown")
        identifiers = [x for x in [namespace, service] if x and x != "unknown"] or ["unknown"]

        return EvidenceCandidate(
            entity_type=EntityType.NAMESPACE,
            raw_identifiers=identifiers,
            signal_source=SignalSource.K8S_STATE,
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
    def _extract_events(content: Any) -> list[dict]:
        if isinstance(content, list):
            return [e for e in content if isinstance(e, dict)]
        if isinstance(content, dict):
            items = content.get("items", [])
            if isinstance(items, list):
                return [e for e in items if isinstance(e, dict)]
        return []

    @staticmethod
    def _extract_non_events(content: Any) -> dict:
        """Extract non-event K8s data (pod status, deployment status, etc.)."""
        if isinstance(content, dict) and "items" not in content:
            return {k: v for k, v in content.items() if k not in ("apiVersion", "kind")}
        return {}
