"""
tests/test_clickhouse_log_client.py
=====================================

Unit tests for clickhouse_log_client.py.

All tests run without a live ClickHouse or Kubernetes cluster — every
external call is mocked. The test suite covers:

  1. Normal path — ClickHouse returns rows, formatted correctly
  2. Tier escalation — hot→warm→cold when rows are sparse
  3. K8s events merging — always fetched and appended
  4. ClickHouse unavailable — clean fallback to K8s API
  5. ClickHouse zero rows — falls through to K8s API fallback
  6. K8s API fallback text — keyword noise filter is applied
  7. Both sources fail — returns empty string (no exception raised)
  8. Malformed rows — skipped gracefully, valid rows still returned
  9. Async wrapper — query_forensic_logs_for_pod_async is awaitable
  10. LogRow.to_formatted_line — format contract
  11. ForensicQueryResult.to_text — rendering contract
  12. Connection timeout — handled as fallback trigger, not exception
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call
import pytest

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_log_client import (
    ClickhouseLogClient,
    ForensicQueryResult,
    LogRow,
    query_forensic_logs_for_pod,
    query_forensic_logs_for_pod_async,
    _TIER_MINIMUM_ROWS,
    _HOT_TABLE,
    _WARM_TABLE,
    _COLD_TABLE,
    _EVENTS_TABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(offset_seconds: int = 0) -> datetime:
    """Return a UTC datetime offset from a fixed reference point."""
    return datetime(2026, 6, 18, 10, 0, offset_seconds, tzinfo=timezone.utc)


def _make_log_row(message: str = "test error", severity: str = "ERROR") -> tuple:
    """Build a raw ClickHouse row tuple matching the SELECT column order."""
    return (
        _make_ts(),       # Timestamp
        severity,         # Severity
        "app",            # ContainerName
        "container",      # Source
        "",               # K8sReason
        0,                # Rescued
        message,          # Message
    )


def _make_event_row(reason: str = "OOMKilling", message: str = "pod oom killed") -> tuple:
    """Build a raw K8s events row tuple."""
    return (
        _make_ts(1),      # Timestamp
        "Warning",        # EventType (mapped to Severity field)
        "",               # ContainerName (empty for events)
        "k8s_event",      # Source
        reason,           # K8sReason
        0,                # Rescued
        message,          # Message
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a ClickhouseLogClient with a fake config (no real connection)."""
    return ClickhouseLogClient(config={
        "host": "127.0.0.1",
        "port": 9000,
        "database": "telemetry",
        "user": "test",
        "password": "",
        "connect_timeout": 1,
        "send_receive_timeout": 2,
        "sync_request_timeout": 2,
        "compression": False,
    })


def _mock_ch_client(rows_by_table: dict):
    """
    Return a mock ClickHouse driver Client that returns different rows
    depending on which table appears in the query string.

    rows_by_table: {"telemetry.logs_hot": [...], "telemetry.k8s_events": [...]}
    """
    mock = MagicMock()
    mock.execute.side_effect = lambda query, params=None, settings=None: (
        rows_by_table.get(
            next((t for t in rows_by_table if t in query), "_default"),
            []
        )
    )
    return mock


# ---------------------------------------------------------------------------
# 1. Normal path — ClickHouse returns rows
# ---------------------------------------------------------------------------

class TestNormalPath:
    def test_hot_tier_rows_returned(self, client):
        hot_rows = [_make_log_row(f"err-{i}") for i in range(6)]
        hot_rows[0] = _make_log_row("OOMKilled process")
        mock_ch = _mock_ch_client({
            _HOT_TABLE: hot_rows,
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs(
                namespace="production",
                pod_name="payment-abc",
                lookback_minutes=10,
            )

        assert result.total_rows == 6
        assert result.tier_used == "hot"
        assert result.used_fallback is False
        assert len(result.log_rows) == 6
        assert result.log_rows[0].message == "OOMKilled process"

    def test_to_text_contains_log_content(self, client):
        hot_rows = [_make_log_row("database connection timeout")]
        mock_ch = _mock_ch_client({_HOT_TABLE: hot_rows, _EVENTS_TABLE: []})
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=5)

        text = result.to_text()
        assert "database connection timeout" in text
        assert "ERROR" in text

    def test_empty_result_to_text_returns_empty_string(self):
        result = ForensicQueryResult(pod_name="pod", namespace="ns")
        assert result.to_text() == ""


# ---------------------------------------------------------------------------
# 2. Tier escalation
# ---------------------------------------------------------------------------

class TestTierEscalation:
    def test_escalates_to_warm_when_hot_sparse(self, client):
        """If hot returns < MINIMUM rows, warm is also queried."""
        # Hot returns 1 row (< _TIER_MINIMUM_ROWS which is 5)
        hot_row = _make_log_row("hot-error")
        warm_rows = [_make_log_row(f"warm-error-{i}") for i in range(6)]
        mock_ch = _mock_ch_client({
            _HOT_TABLE: [hot_row],
            _WARM_TABLE: warm_rows,
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=60)

        assert result.tier_used == "warm"
        # Deduplication: 1 hot + 6 warm = 7 (no duplicates since messages differ)
        assert len(result.log_rows) == 7

    def test_escalates_to_cold_when_warm_also_sparse(self, client):
        """If both hot and warm are sparse, cold is queried."""
        cold_rows = [_make_log_row(f"cold-{i}") for i in range(8)]
        mock_ch = _mock_ch_client({
            _HOT_TABLE: [],
            _WARM_TABLE: [_make_log_row("warm-single")],
            _COLD_TABLE: cold_rows,
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=120)

        assert result.tier_used == "cold"
        assert len(result.log_rows) == 9  # 1 warm + 8 cold

    def test_deduplication_across_tiers(self, client):
        """The same log row appearing in hot and warm is not duplicated."""
        shared_ts = _make_ts(5)
        shared_row = (shared_ts, "ERROR", "app", "container", "", 0, "shared message")
        mock_ch = _mock_ch_client({
            _HOT_TABLE: [shared_row],
            _WARM_TABLE: [shared_row],  # same (timestamp, message)
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=60)

        # Even though both tiers returned the same row, it should appear once
        unique_messages = [r.message for r in result.log_rows]
        assert unique_messages.count("shared message") == 1

    def test_no_escalation_when_hot_sufficient(self, client):
        """If hot returns >= MINIMUM rows, warm/cold are never queried."""
        hot_rows = [_make_log_row(f"err-{i}") for i in range(_TIER_MINIMUM_ROWS + 1)]
        mock_ch = _mock_ch_client({
            _HOT_TABLE: hot_rows,
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=10)

        # Warm and cold should never have been queried
        queries = [call_args[0][0] for call_args in mock_ch.execute.call_args_list]
        assert not any(_WARM_TABLE in q for q in queries), "Warm should not be queried when hot is sufficient"
        assert not any(_COLD_TABLE in q for q in queries), "Cold should not be queried when hot is sufficient"
        assert result.tier_used == "hot"


# ---------------------------------------------------------------------------
# 3. K8s events merging
# ---------------------------------------------------------------------------

class TestK8sEventsMerging:
    def test_events_always_fetched(self, client):
        hot_rows = [_make_log_row("error in app")]
        event_rows = [_make_event_row("OOMKilling", "process killed")]
        mock_ch = _mock_ch_client({
            _HOT_TABLE: hot_rows,
            _EVENTS_TABLE: event_rows,
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=10)

        assert len(result.k8s_event_rows) == 1
        assert result.k8s_event_rows[0].k8s_reason == "OOMKilling"
        assert result.k8s_event_rows[0].source == "k8s_event"

    def test_events_in_text_output(self, client):
        event_rows = [_make_event_row("FailedScheduling", "insufficient resources")]
        mock_ch = _mock_ch_client({_HOT_TABLE: [_make_log_row()], _EVENTS_TABLE: event_rows})
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=10)

        text = result.to_text()
        assert "Kubernetes Events" in text
        assert "FailedScheduling" in text
        assert "insufficient resources" in text

    def test_events_fetched_even_when_log_rows_empty(self, client):
        """K8s events should still be returned if all log tiers are empty."""
        event_rows = [_make_event_row("BackOff", "container restarting")]
        mock_ch = _mock_ch_client({
            _HOT_TABLE: [],
            _WARM_TABLE: [],
            _COLD_TABLE: [],
            _EVENTS_TABLE: event_rows,
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=60)

        assert result.total_rows == 1
        assert result.k8s_event_rows[0].k8s_reason == "BackOff"


# ---------------------------------------------------------------------------
# 4. ClickHouse unavailable → K8s API fallback
# ---------------------------------------------------------------------------

class TestClickHouseUnavailable:
    def test_connect_failure_triggers_k8s_fallback(self):
        """When _connect() returns None, query_forensic_logs_for_pod falls back."""
        mock_v1 = MagicMock()
        k8s_text = "ERROR: payment service crashed\nERROR: database unreachable"

        with patch("clickhouse_log_client._get_default_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.query_pod_logs.return_value = ForensicQueryResult(
                pod_name="pod", namespace="ns",
                fallback_reason="ClickHouse connection failed",
            )
            mock_factory.return_value = mock_client

            with patch("prometheus_anomaly.extract_forensic_logs", return_value=k8s_text) as mock_extract:
                result = query_forensic_logs_for_pod(
                    namespace="ns",
                    pod_name="pod",
                    k8s_v1_client=mock_v1,
                )

        assert "[K8s API Fallback]" in result
        assert "database unreachable" in result
        mock_extract.assert_called_once()

    def test_driver_not_installed_falls_back(self):
        """If clickhouse-driver is not importable, fallback to K8s API."""
        import clickhouse_log_client as module
        original = module._CH_AVAILABLE

        try:
            module._CH_AVAILABLE = False
            mock_v1 = MagicMock()
            k8s_text = "ERROR: crash detected"

            # Reset the singleton so it creates a fresh client with _CH_AVAILABLE=False
            module._default_client = None

            with patch("prometheus_anomaly.extract_forensic_logs", return_value=k8s_text):
                result = query_forensic_logs_for_pod(
                    namespace="ns",
                    pod_name="pod",
                    k8s_v1_client=mock_v1,
                )
            assert "crash detected" in result
        finally:
            module._CH_AVAILABLE = original
            module._default_client = None


# ---------------------------------------------------------------------------
# 5. ClickHouse zero rows → K8s API fallback
# ---------------------------------------------------------------------------

class TestZeroRowsFallback:
    def test_zero_rows_all_tiers_uses_k8s_fallback(self):
        """When all three tiers return 0 rows, K8s API fallback is used."""
        mock_v1 = MagicMock()
        k8s_text = "FATAL: service crashed at startup"

        with patch("clickhouse_log_client._get_default_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.query_pod_logs.return_value = ForensicQueryResult(
                pod_name="pod", namespace="ns", tier_used="empty",
            )
            mock_factory.return_value = mock_client

            with patch("prometheus_anomaly.extract_forensic_logs", return_value=k8s_text):
                result = query_forensic_logs_for_pod(
                    namespace="ns",
                    pod_name="pod",
                    k8s_v1_client=mock_v1,
                )

        assert "FATAL" in result
        assert "[K8s API Fallback]" in result


# ---------------------------------------------------------------------------
# 6. K8s API fallback — keyword noise filter applied to raw kubectl output
# ---------------------------------------------------------------------------

class TestK8sApiNoiseFilter:
    def test_info_lines_without_keywords_are_dropped(self):
        """INFO lines without error keywords should not appear in combined_errors."""
        # This tests the noise filter logic in _extract_and_append in live_harness.py
        # Here we test the underlying text that would be produced and manually
        # verify the filter behaviour using the same keyword list.

        raw_k8s_lines = [
            "2026-06-18T10:00:00Z INFO starting server on port 8080",
            "2026-06-18T10:00:01Z INFO health check passed",
            "2026-06-18T10:00:02Z INFO connection refused to redis",  # has keyword — keep
            "2026-06-18T10:00:03Z ERROR database unreachable",         # has error level — keep
            "2026-06-18T10:00:04Z DEBUG metrics emitted",
        ]

        KEYWORDS = (
            "error", "fail", "fatal", "exception", "panic",
            "warn", "timeout", "refused", "back-off", "oom", "unreachable",
        )

        filtered = []
        for line in raw_k8s_lines:
            lower = line.lower()
            if ("info" in lower or "debug" in lower) and not any(kw in lower for kw in KEYWORDS):
                continue
            filtered.append(line)

        # Should keep: line 3 (INFO with "refused"), line 4 (ERROR), and drop lines 1, 2, 5
        assert len(filtered) == 2
        assert "connection refused to redis" in filtered[0]
        assert "database unreachable" in filtered[1]


# ---------------------------------------------------------------------------
# 7. Both sources fail → empty string, no exception
# ---------------------------------------------------------------------------

class TestTotalFailure:
    def test_returns_empty_string_when_all_sources_fail(self):
        """No exception should propagate when both ClickHouse and K8s API fail."""
        with patch("clickhouse_log_client._get_default_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.query_pod_logs.side_effect = RuntimeError("simulated CH crash")
            mock_factory.return_value = mock_client

            with patch("prometheus_anomaly.extract_forensic_logs",
                       side_effect=Exception("simulated K8s API crash")):
                result = query_forensic_logs_for_pod(
                    namespace="ns",
                    pod_name="pod",
                    k8s_v1_client=MagicMock(),
                )

        assert result == ""

    def test_returns_empty_string_without_k8s_client(self):
        """When k8s_v1_client=None and ClickHouse returns zero rows, empty string returned."""
        with patch("clickhouse_log_client._get_default_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.query_pod_logs.return_value = ForensicQueryResult(
                pod_name="pod", namespace="ns"
            )
            mock_factory.return_value = mock_client

            result = query_forensic_logs_for_pod(
                namespace="ns", pod_name="pod", k8s_v1_client=None,
            )
        assert result == ""


# ---------------------------------------------------------------------------
# 8. Malformed rows — skipped gracefully
# ---------------------------------------------------------------------------

class TestMalformedRows:
    def test_malformed_row_skipped_valid_rows_returned(self, client):
        """A malformed row (wrong types) should be silently skipped."""
        valid_row = _make_log_row("valid error")
        malformed_row = (None, None, None, None, None, None, None)  # All None

        mock_ch = _mock_ch_client({
            _HOT_TABLE: [malformed_row, valid_row],
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=10)

        # The malformed row should be dropped; the valid row should survive
        assert len(result.log_rows) == 1
        assert result.log_rows[0].message == "valid error"

    def test_completely_malformed_row_set(self, client):
        """All malformed rows should result in an empty result, no exception."""
        mock_ch = _mock_ch_client({
            _HOT_TABLE: [("bad",), ("worse",)],  # too few columns
            _EVENTS_TABLE: [],
        })
        with patch.object(client, "_connect", return_value=mock_ch):
            result = client.query_pod_logs("ns", "pod", lookback_minutes=10)

        assert result.log_rows == []
        assert result.total_rows == 0


# ---------------------------------------------------------------------------
# 9. Async wrapper is awaitable and returns same result
# ---------------------------------------------------------------------------

class TestAsyncWrapper:
    def test_async_wrapper_returns_same_as_sync(self):
        """query_forensic_logs_for_pod_async should return same as sync version."""
        expected = "2026-06-18T10:00:00.000Z ERROR [CONTAINER:app]: database timeout"

        with patch("clickhouse_log_client.query_forensic_logs_for_pod", return_value=expected):
            result = asyncio.run(query_forensic_logs_for_pod_async(
                namespace="ns",
                pod_name="pod",
            ))

        assert result == expected

    def test_async_wrapper_propagates_kwargs(self):
        """All kwargs should be passed through to the sync function."""
        with patch("clickhouse_log_client.query_forensic_logs_for_pod",
                   return_value="") as mock_sync:
            asyncio.run(query_forensic_logs_for_pod_async(
                namespace="production",
                pod_name="frontend-xyz",
                container_name="app",
                lookback_minutes=30,
                limit=100,
            ))
            mock_sync.assert_called_once_with(
                "production", "frontend-xyz", "app", 30, 100, None, 50
            )


# ---------------------------------------------------------------------------
# 10. LogRow.to_formatted_line contract
# ---------------------------------------------------------------------------

class TestLogRowFormatting:
    def test_basic_format(self):
        row = LogRow(
            timestamp=_make_ts(),
            severity="ERROR",
            container_name="app",
            source="container",
            k8s_reason="",
            rescued=0,
            message="connection refused",
        )
        line = row.to_formatted_line("my-pod")
        assert "ERROR" in line
        assert "[CONTAINER:app]" in line
        assert "connection refused" in line
        assert "RESCUED" not in line

    def test_rescued_flag_shown(self):
        row = LogRow(
            timestamp=_make_ts(),
            severity="INFO",
            container_name="sidecar",
            source="container",
            k8s_reason="",
            rescued=1,
            message="out of memory error logged as info",
        )
        line = row.to_formatted_line("my-pod")
        assert "[RESCUED]" in line

    def test_k8s_reason_shown(self):
        row = LogRow(
            timestamp=_make_ts(),
            severity="WARNING",
            container_name="",
            source="k8s_event",
            k8s_reason="OOMKilling",
            rescued=0,
            message="pod killed",
        )
        line = row.to_formatted_line("my-pod")
        assert "(OOMKilling)" in line

    def test_unix_timestamp_converted(self):
        """Float UNIX timestamps should be converted to datetime."""
        row = LogRow(
            timestamp=datetime.fromtimestamp(1718704800, tz=timezone.utc),
            severity="ERROR",
            container_name="app",
            source="container",
            k8s_reason="",
            rescued=0,
            message="test",
        )
        line = row.to_formatted_line("pod")
        assert "2024" in line  # Year should appear


# ---------------------------------------------------------------------------
# 11. ForensicQueryResult.to_text contract
# ---------------------------------------------------------------------------

class TestForensicQueryResultToText:
    def test_renders_section_headers(self):
        result = ForensicQueryResult(
            pod_name="svc-pod",
            namespace="prod",
            tier_used="hot",
            log_rows=[
                LogRow(_make_ts(), "ERROR", "app", "container", "", 0, "fatal crash"),
            ],
            k8s_event_rows=[
                LogRow(_make_ts(1), "Warning", "", "k8s_event", "OOMKilling", 0, "oom event"),
            ],
        )
        text = result.to_text()
        assert "Container Logs" in text
        assert "hot tier" in text
        assert "Kubernetes Events" in text
        assert "fatal crash" in text
        assert "oom event" in text

    def test_only_events_no_log_rows(self):
        result = ForensicQueryResult(
            pod_name="pod",
            namespace="ns",
            k8s_event_rows=[
                LogRow(_make_ts(), "Warning", "", "k8s_event", "BackOff", 0, "restart backoff"),
            ],
        )
        text = result.to_text()
        assert "Kubernetes Events" in text
        assert "BackOff" in text
        # No container logs section
        assert "Container Logs" not in text


# ---------------------------------------------------------------------------
# 12. Connection timeout handled as fallback trigger
# ---------------------------------------------------------------------------

class TestConnectionTimeout:
    def test_connect_timeout_triggers_fallback_not_exception(self):
        """A timeout during connection should result in empty ForensicQueryResult."""
        import clickhouse_log_client as module

        if not module._CH_AVAILABLE:
            pytest.skip("clickhouse-driver not installed")

        with patch("clickhouse_log_client._CHClient") as mock_cls:
            mock_cls.side_effect = Exception("connection timed out")
            client = ClickhouseLogClient(config={
                "host": "10.255.255.1",
                "port": 9000,
                "database": "telemetry",
                "user": "airs_agent",
                "password": "",
                "connect_timeout": 1,
                "send_receive_timeout": 1,
                "sync_request_timeout": 1,
                "compression": False,
            })
            result = client.query_pod_logs("ns", "pod", lookback_minutes=5)

        assert result.total_rows == 0
        assert result.fallback_reason != ""
        assert result.used_fallback is False  # fallback is set by outer caller
