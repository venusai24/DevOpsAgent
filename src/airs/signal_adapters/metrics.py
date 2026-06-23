"""
Metrics Signal Adapter (Prometheus) — Module 1.6.

Handles METRICS signal type: translates intents to Prometheus MCP calls,
applies Z-score pre-filtering, and normalizes to EvidenceCandidate.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from airs.models.evidence import EvidenceCandidate, EntityType, SignalSource
from airs.models.intents import ToolSpec, SignalType
from airs.signal_adapters.base import (
    FilteredSignalPayload,
    RawSignalPayload,
    SignalAdapter,
)
from airs.signal_adapters.prefilters import zscore_filter

log = logging.getLogger(__name__)


class MetricsSignalAdapter(SignalAdapter):
    """
    Prometheus metrics signal adapter.

    execute:     Calls Prometheus MCP tools (execute_query / execute_range_query).
    pre_filter:  Z-score anomaly detection on time-series results.
    normalize:   Produces EvidenceCandidate with entity classification.
    """

    @property
    def signal_type(self) -> SignalType:
        return SignalType.METRICS

    async def execute(
        self,
        tool_spec: ToolSpec,
        mcp_client: Any,
    ) -> RawSignalPayload:
        spec = tool_spec
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
            idempotency_key="",  # Set by activity
            arguments_hash="",   # Set by activity
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

        content = payload.content or {}

        # Prometheus range query → list of (timestamp, value) pairs
        data_points = self._extract_prometheus_data_points(content)

        if data_points:
            filtered_points, stats = zscore_filter(
                data_points, value_key="value", z_threshold=2.0
            )
        else:
            filtered_points = []
            stats = {"method": "passthrough", "reason": "no_data_points"}

        # Extract key summary statistics
        metric_name = self._extract_metric_name(content)
        summary_stats = self._compute_summary_stats(
            [float(dp["value"]) for dp in data_points if "value" in dp]
        )

        filtered_content = {
            "metric_name": metric_name,
            "service_filter": incident_service,
            "namespace_filter": incident_namespace,
            "summary_stats": summary_stats,
            "anomalous_points": filtered_points[:10],  # Cap at 10 for context budget
            "total_data_points": len(data_points),
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

    def normalize(
        self,
        filtered: FilteredSignalPayload,
        hop_index: int,
    ) -> EvidenceCandidate:
        content = filtered.filtered_content
        metric_name = content.get("metric_name", "unknown_metric")
        service = content.get("service_filter", "unknown")

        # Classify entity type based on metric name
        entity_type = self._classify_entity_type(metric_name)
        identifiers = [service] if service else [metric_name]

        return EvidenceCandidate(
            entity_type=entity_type,
            raw_identifiers=identifiers,
            signal_source=SignalSource.METRICS,
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

    # ─── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_prometheus_data_points(content: Any) -> list[dict]:
        """Extract (timestamp, value) pairs from Prometheus JSON response."""
        try:
            # Prometheus range query format: {data: {result: [{values: [[ts, val], ...]}]}}
            result = content.get("data", {}).get("result", [])
            if not result:
                # Instant query format: values are in the metric dict
                return []
            points = []
            for series in result:
                for ts, val in series.get("values", []):
                    try:
                        points.append({"timestamp": float(ts), "value": float(val)})
                    except (TypeError, ValueError):
                        pass
            return points
        except (AttributeError, TypeError):
            return []

    @staticmethod
    def _extract_metric_name(content: Any) -> str:
        """Extract the metric name from Prometheus response."""
        try:
            result = content.get("data", {}).get("result", [])
            if result:
                return result[0].get("metric", {}).get("__name__", "unknown")
        except (AttributeError, TypeError):
            pass
        return "unknown_metric"

    @staticmethod
    def _compute_summary_stats(values: list[float]) -> dict[str, float]:
        """Compute mean, p50, p95, p99, min, max for a list of values."""
        if not values:
            return {}
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(p: float) -> float:
            idx = int(p * (n - 1) / 100)
            return sorted_vals[idx]

        return {
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": round(sum(values) / n, 4),
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
            "count": n,
        }

    @staticmethod
    def _classify_entity_type(metric_name: str) -> EntityType:
        """Classify entity type from metric name prefix/keywords."""
        m = metric_name.lower()
        if any(k in m for k in ["http", "grpc", "request", "response"]):
            return EntityType.SERVICE
        if any(k in m for k in ["container", "pod", "kubelet"]):
            return EntityType.POD
        if any(k in m for k in ["node", "machine", "cpu", "memory"]):
            return EntityType.NODE
        if any(k in m for k in ["redis", "cache", "memcached"]):
            return EntityType.CACHE
        if any(k in m for k in ["pg", "postgres", "mysql", "mongo", "db"]):
            return EntityType.DATABASE
        return EntityType.SERVICE
