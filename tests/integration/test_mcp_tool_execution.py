"""
Integration Tests — MCP Tool Execution Pipeline (Module 1.12).

Tests the signal adapter execute → pre_filter → normalize pipeline
using mock MCP clients and synthetic responses. No real MCP server needed.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from airs.mcp.client import MCPToolResponse
from airs.models.evidence import EntityType, SignalSource
from airs.models.intents import ExecutionIntent, IntentAction, SignalType, ToolSpec, ToolTier
from airs.signal_adapters import get_adapter


def make_mock_client(content: dict) -> MagicMock:
    """Create a mock MCPToolClient returning the given content."""
    client = MagicMock()
    client.invoke = AsyncMock(return_value=MCPToolResponse(
        success=True,
        content=content,
        raw_text=json.dumps(content),
        raw_digest="a" * 64,
        error_code=None,
        error_message=None,
        server_id="prometheus",
        tool_name="execute_range_query",
        latency_ms=50.0,
    ))
    return client


class TestMetricsAdapterPipeline:
    @pytest.mark.asyncio
    async def test_execute_returns_raw_payload(
        self, prometheus_range_query_response
    ):
        """Adapter.execute() returns a RawSignalPayload from MCP response."""
        adapter = get_adapter(SignalType.METRICS)
        mock_client = make_mock_client(prometheus_range_query_response)

        intent = ExecutionIntent(
            action=IntentAction.EXECUTE_TOOL,
            tool_spec=ToolSpec(
                tool_name="execute_range_query",
                mcp_server_id="prometheus",
                arguments={"query": "rate(http_requests_total[5m])"},
                tier=ToolTier.OBSERVATION,
                signal_type=SignalType.METRICS,
            ),
            reasoning="Testing",
            missing_mass_at_decision=0.8,
        )

        raw = await adapter.execute(intent, mock_client)
        assert raw.success is True
        assert raw.server_id == "prometheus"

    def test_prefilter_extracts_anomalous_points(
        self, prometheus_range_query_response
    ):
        """Pre-filter identifies the spike in p99 latency."""
        from airs.signal_adapters.base import RawSignalPayload

        adapter = get_adapter(SignalType.METRICS)
        raw = RawSignalPayload(
            server_id="prometheus",
            tool_name="execute_range_query",
            content=prometheus_range_query_response,
            raw_text=json.dumps(prometheus_range_query_response),
            raw_digest="a" * 64,
            idempotency_key="idem-test",
            arguments_hash="hash-test",
            success=True,
        )

        filtered = adapter.pre_filter(
            raw,
            incident_service="payment-service",
            incident_namespace="production",
        )

        assert filtered.filtered_content["metric_name"] != ""
        # Spike at 2.5s should be detected as anomalous
        anomalous = filtered.filtered_content.get("anomalous_points", [])
        assert len(anomalous) > 0
        # The 2.5 spike should be in the anomalous set
        values = [float(p["value"]) for p in anomalous]
        assert any(v > 1.0 for v in values)

    def test_normalize_returns_evidence_candidate(
        self, prometheus_range_query_response
    ):
        """Normalize produces a valid EvidenceCandidate."""
        from airs.signal_adapters.base import FilteredSignalPayload

        adapter = get_adapter(SignalType.METRICS)
        filtered = FilteredSignalPayload(
            server_id="prometheus",
            tool_name="execute_range_query",
            filtered_content={
                "metric_name": "http_request_duration_seconds",
                "service_filter": "payment-service",
                "anomalous_points": [{"timestamp": 1718000120.0, "value": 2.5}],
                "summary_stats": {"mean": 0.12, "p99": 2.5},
                "total_data_points": 5,
            },
            token_estimate=180,
            filter_stats={"method": "zscore"},
            raw_digest="a" * 64,
            idempotency_key="idem-test",
            arguments_hash="hash-test",
            received_at=datetime.now(timezone.utc),
        )

        candidate = adapter.normalize(filtered, hop_index=1)

        assert candidate.signal_source == SignalSource.METRICS
        assert candidate.hop_index == 1
        assert candidate.token_count == 180
        assert "payment-service" in candidate.raw_identifiers


class TestLogsAdapterPipeline:
    def test_log_dedup_collapses_repeated_errors(self, loki_log_response):
        """Log dedup collapses 3 'Connection refused' lines to 1 pattern."""
        from airs.signal_adapters.base import RawSignalPayload

        adapter = get_adapter(SignalType.LOGS)
        raw = RawSignalPayload(
            server_id="loki",
            tool_name="loki_query",
            content=loki_log_response,
            raw_text=json.dumps(loki_log_response),
            raw_digest="b" * 64,
            idempotency_key="idem-logs",
            arguments_hash="hash-logs",
            success=True,
        )

        filtered = adapter.pre_filter(raw, incident_service="payment-service")
        patterns = filtered.filtered_content.get("log_patterns", [])

        # 5 raw lines → 3-4 deduplicated patterns
        total_raw = filtered.filtered_content.get("total_raw_lines", 0)
        assert total_raw == 5
        # Patterns should be fewer than raw lines due to dedup
        assert len(patterns) < total_raw


class TestTracesAdapterPipeline:
    def test_error_span_extraction(self, jaeger_trace_response):
        """Error spans (status=503) are extracted from trace."""
        from airs.signal_adapters.base import RawSignalPayload

        adapter = get_adapter(SignalType.TRACES)
        raw = RawSignalPayload(
            server_id="jaeger",
            tool_name="search_traces",
            content=jaeger_trace_response,
            raw_text=json.dumps(jaeger_trace_response),
            raw_digest="c" * 64,
            idempotency_key="idem-traces",
            arguments_hash="hash-traces",
            success=True,
        )

        filtered = adapter.pre_filter(raw, incident_service="payment-service")
        error_spans = filtered.filtered_content.get("error_spans", [])

        # 2 of 3 spans have errors
        assert len(error_spans) == 2

    def test_failed_mcp_response_handled_gracefully(self):
        """Failed MCP response produces error-flagged FilteredSignalPayload."""
        from airs.signal_adapters.base import RawSignalPayload

        adapter = get_adapter(SignalType.TRACES)
        raw = RawSignalPayload(
            server_id="jaeger",
            tool_name="search_traces",
            content=None,
            raw_text="",
            raw_digest="",
            idempotency_key="idem-fail",
            arguments_hash="hash-fail",
            success=False,
            error_code=503,
            error_message="Jaeger unreachable",
        )

        filtered = adapter.pre_filter(raw)
        assert filtered.filtered_content.get("success") is False
        assert "error" in filtered.filtered_content
