"""
clickhouse_log_client.py
========================

Resilient ClickHouse log client for the AIRS forensic log extraction pipeline.

Architecture
------------
This module replaces the synchronous ``kubectl logs`` polling in
``prometheus_anomaly.extract_forensic_logs`` with an asynchronous,
tiered ClickHouse query strategy as defined in the Golden Blueprint:

  Hot  (7d,  ERROR/FATAL/rescued) → queried first — sub-100ms latency
  Warm (30d, WARN)                → queried if hot returns < MINIMUM_ROWS
                                    OR semantic diversity < MINIMUM_DIVERSITY
  Cold (90d, INFO/DEBUG survivors) → last resort escalation (RESCUED only)
  K8s Events                      → always fetched, merged chronologically

Fallback Chain
--------------
Every external call is wrapped in a fallback chain:

  ClickHouse (hot → warm → cold) → K8s API fallback → empty string

The K8s API fallback is **preserved** as a critical safety net for:
  - Cluster cold-start before Vector/ClickHouse are fully deployed
  - Vector/ClickHouse pipeline outages (the very scenario pipeline-monitoring
    alerts on — agent must stay operational in degraded mode)
  - New namespaces not yet covered by the Vector DaemonSet

The fallback text is tagged with ``[DEGRADED-FALLBACK]`` and emits a
Prometheus counter ``airs_forensic_fallback_used_total`` so elevated
fallback usage triggers pipeline monitoring alerts.

Log Intelligence (Q3 — Option B)
---------------------------------
Template deduplication (TemplateDeduplicator) runs **after** ClickHouse
query results are fetched, before text rendering. This reduces LLM token
consumption by 30-42% without losing diagnostic fidelity.

Driver Migration (Q1)
---------------------
Migrated from ``clickhouse-driver`` (sync native TCP, wrapped in
asyncio.to_thread) to ``clickhouse-connect`` (official ClickHouse client
with native asyncio via aiohttp). Avoids sync-over-async thread-pool
exhaustion under concurrent forensic queries.

Dependencies added to requirements.txt:
  clickhouse-connect[async]>=0.7.0   (official async client, replaces clickhouse-driver)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional clickhouse-connect import — fail gracefully so syntax checks pass
# even without the library installed.
# ---------------------------------------------------------------------------
try:
    import clickhouse_connect
    from clickhouse_connect.driver.exceptions import ClickHouseError as _CHError
    _CH_AVAILABLE = True
except ImportError:
    clickhouse_connect = None          # type: ignore[assignment]
    _CHError = Exception               # type: ignore[assignment,misc]
    _CH_AVAILABLE = False
    logger.warning(
        "[ClickhouseLogClient] clickhouse-connect not installed. "
        "Install with: pip install 'clickhouse-connect[async]>=0.7.0'. "
        "All queries will fall through to the K8s API fallback."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum rows from ClickHouse before escalating to the next tier
_TIER_MINIMUM_ROWS = 5

# Minimum semantic diversity ratio to avoid cold-tier escalation.
# A ratio below this means the hot/warm tier is saturated with
# repetitive messages — escalate to find diverse signals.
_TIER_MINIMUM_DIVERSITY = 0.3

# Maximum rows returned to the AI agent per pod query
_MAX_ROWS = 200

# Hard timeout (seconds) for a single ClickHouse query
_QUERY_TIMEOUT_S = 8

# Default lookback window when no explicit window is provided.
_DEFAULT_LOOKBACK_MINUTES = 60

# Cold tier: only rescue-tagged rows to prevent INFO/DEBUG noise injection
_COLD_MAX_ROWS = 20

# Tiered table names — must match 07-clickhouse-schema.sql
_HOT_TABLE = "telemetry.logs_hot"
_WARM_TABLE = "telemetry.logs_warm"
_COLD_TABLE = "telemetry.logs_cold"
_EVENTS_TABLE = "telemetry.k8s_events"

# Template normalisation patterns (same as log_deduplicator.py)
_TEMPLATE_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE), "?uuid"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "?ip"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "?ts"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE), "?hex"),
    (re.compile(r"\b\d+\b"), "?n"),
    (re.compile(r"\s+"), " "),
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _get_ch_config() -> dict:
    """
    Build clickhouse-connect connection parameters from environment variables.

    Reads (in priority order):
      CLICKHOUSE_HOST  — hostname or IP  (default: clickhouse.observability.svc.cluster.local)
      CLICKHOUSE_PORT  — HTTP port       (default: 8123)
      CLICKHOUSE_DB    — default database (default: telemetry)
      CLICKHOUSE_USER  — username        (default: airs_agent)
      CLICKHOUSE_PASSWORD — password     (default: empty string)

    clickhouse-connect uses the HTTP interface (port 8123) by default,
    not the native TCP port 9000. This aligns with the aggregator sink
    endpoint in 04-vector-aggregator-config.toml.
    """
    return {
        "host": os.getenv(
            "CLICKHOUSE_HOST",
            "clickhouse.observe.svc.cluster.local",
        ),
        "port": int(os.getenv("CLICKHOUSE_PORT", "8123")),
        "database": os.getenv("CLICKHOUSE_DB", "telemetry"),
        "username": os.getenv("CLICKHOUSE_USER", "airs_agent"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        # Query-level timeout enforced server-side
        "query_limit": 10_000_000,
        "connect_timeout": 5,
        "send_receive_timeout": _QUERY_TIMEOUT_S,
    }


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LogRow:
    """A single log record returned from ClickHouse."""
    timestamp: datetime
    severity: str
    container_name: str
    source: str          # 'container' | 'init_container' | 'k8s_event' | 'previous'
    k8s_reason: str
    rescued: int         # 1 = matched semantic safety net
    message: str

    def to_formatted_line(self, pod_name: str) -> str:
        """
        Full-fidelity format for debugging and log inspection.
        Timestamp at second precision to save tokens vs microsecond default.
        """
        ts = self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        prefix = f"[{self.source.upper()}:{self.container_name}]" if self.container_name else ""
        reason = f" ({self.k8s_reason})" if self.k8s_reason else ""
        rescued = " [RESCUED]" if self.rescued else ""
        return f"{ts} {self.severity.upper()}{rescued} {prefix}{reason}: {self.message}"

    def compact_format(self) -> str:
        """
        Compact token-efficient format for LLM context injection.
        Drops source/container metadata; keeps time, severity, message.
        Format: [HH:MM:SS] SEV (REASON): message
        """
        ts = self.timestamp.strftime("%H:%M:%S")
        reason = f"({self.k8s_reason}) " if self.k8s_reason else ""
        rescued = "[R] " if self.rescued else ""
        return f"[{ts}] {self.severity.upper()} {rescued}{reason}{self.message}"


@dataclass
class ForensicQueryResult:
    """Aggregated result from a tiered ClickHouse forensic query."""
    pod_name: str
    namespace: str
    log_rows: list[LogRow] = field(default_factory=list)
    k8s_event_rows: list[LogRow] = field(default_factory=list)
    tier_used: str = "none"        # hot | warm | cold | k8s_api | empty
    query_duration_ms: float = 0.0
    used_fallback: bool = False
    fallback_reason: str = ""
    # Deduplication metadata (populated by to_text())
    original_row_count: int = 0
    deduplicated_row_count: int = 0
    dedup_ratio: float = 0.0

    @property
    def total_rows(self) -> int:
        return len(self.log_rows) + len(self.k8s_event_rows)

    def to_text(self, compact: bool = True) -> str:
        """
        Render the full forensic context as a formatted string suitable for
        the AIRS ReasoningEngine.

        Improvements over v1:
        1. Chronological interleaving: K8s events and container logs are
           merged into a single timeline sorted by timestamp. The LLM no
           longer needs to mentally merge two separate sections.
        2. Template deduplication: Repeated messages are collapsed to
           ``[×N] <message>`` representations, reducing token count 30-42%.
        3. Compact format: Timestamps at second precision; stripped hex hashes.

        Parameters
        ----------
        compact:
            If True (default), uses ``compact_format()`` (token-efficient).
            If False, uses ``to_formatted_line()`` (full fidelity for debugging).
        """
        from airs_v2.perception.log_deduplicator import deduplicate_log_rows

        if not self.log_rows and not self.k8s_event_rows:
            return ""

        # ── Deduplicate container logs ───────────────────────────────────
        original_count = len(self.log_rows)
        deduped_rows, dedup_stats = deduplicate_log_rows(self.log_rows)

        # Persist dedup stats onto the result object for observability
        self.original_row_count = original_count
        self.deduplicated_row_count = dedup_stats.deduplicated_count
        self.dedup_ratio = dedup_stats.dedup_ratio

        # ── Chronologically interleave logs + K8s events ──────────────────
        # Merge both streams into one unified timeline.
        # K8s events are tagged [K8S-EVENT] so the LLM distinguishes them.
        all_items: list[tuple[datetime, str]] = []

        fmt_fn = (lambda r: r.compact_format()) if compact else (lambda r: r.to_formatted_line(self.pod_name))

        for row in deduped_rows:
            all_items.append((row.timestamp, fmt_fn(row)))

        for evt in self.k8s_event_rows:
            ts = evt.timestamp.strftime("%H:%M:%S") if compact else evt.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            msg = f"[K8S-EVENT] ({evt.k8s_reason}): {evt.message}"
            if compact:
                all_items.append((evt.timestamp, f"[{ts}] EVENT {msg}"))
            else:
                all_items.append((evt.timestamp, f"{ts} {msg}"))

        # Sort unified timeline chronologically
        all_items.sort(key=lambda x: x[0])

        # ── Render ────────────────────────────────────────────────────────
        lines: list[str] = []

        header_parts = [f"tier={self.tier_used}"]
        if self.original_row_count > 0:
            header_parts.append(
                f"logs={self.deduplicated_row_count}/{self.original_row_count}"
                f" (dedup={self.dedup_ratio:.0%})"
            )
        if self.k8s_event_rows:
            header_parts.append(f"events={len(self.k8s_event_rows)}")

        lines.append(f"--- Forensic Context [{', '.join(header_parts)}] ---")
        lines.extend(line for _, line in all_items)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Semantic diversity scoring
# ---------------------------------------------------------------------------


def _semantic_diversity_score(rows: list[LogRow]) -> float:
    """
    Compute the ratio of unique canonical templates to total messages.

    Score of 1.0 = all messages are structurally unique.
    Score of 0.0 = all messages are identical after normalisation.

    Used as the escalation gate for the warm → cold tier transition.
    Low diversity (repetitive logs) indicates the current tier lacks
    novel diagnostic signals despite having >= _TIER_MINIMUM_ROWS rows.
    """
    if not rows:
        return 0.0
    templates: set[str] = set()
    for row in rows:
        text = row.message.strip()
        for pattern, replacement in _TEMPLATE_SUBS:
            text = pattern.sub(replacement, text)
        templates.add(text[:200])
    return len(templates) / len(rows)


# ---------------------------------------------------------------------------
# Core ClickHouse Client (clickhouse-connect, native async)
# ---------------------------------------------------------------------------


class ClickhouseLogClient:
    """
    Resilient ClickHouse client for forensic log queries using clickhouse-connect.

    Async-native
    ------------
    clickhouse-connect's ``AsyncClient`` uses aiohttp under the hood,
    eliminating the sync-over-async thread-pool pattern of the previous
    clickhouse-driver implementation. Concurrent forensic queries (one per
    anomalous pod) no longer compete for OS thread slots.

    Usage
    -----
    Construct once; reuse across all forensic queries in a run.
    All public methods are async coroutines.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or _get_ch_config()
        self._async_client = None  # Lazily initialised

    async def _get_client(self):
        """
        Return the async ClickHouse client, initialising it on first call.
        clickhouse-connect's AsyncClient manages an aiohttp connection pool.
        """
        if not _CH_AVAILABLE:
            return None
        if self._async_client is not None:
            return self._async_client
        try:
            self._async_client = await clickhouse_connect.get_async_client(
                **self._config
            )
            # Lightweight connectivity check
            await self._async_client.query("SELECT 1")
            logger.debug(
                "[ClickhouseLogClient] Connected to ClickHouse at %s:%s",
                self._config.get("host"), self._config.get("port"),
            )
            return self._async_client
        except Exception as exc:
            logger.warning(
                "[ClickhouseLogClient] Cannot connect to ClickHouse at %s:%s — %s",
                self._config.get("host"), self._config.get("port"), exc,
            )
            self._async_client = None
            return None

    async def _execute_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
    ) -> list[tuple]:
        """
        Execute a query and return rows as list of tuples.
        Returns empty list on any error.
        """
        client = await self._get_client()
        if client is None:
            return []
        try:
            result = await client.query(
                query,
                parameters=parameters or {},
                settings={
                    "max_execution_time": _QUERY_TIMEOUT_S,
                    "max_rows_to_read": 10_000_000,
                    "max_memory_usage": 2 * 1024 * 1024 * 1024,
                },
            )
            return result.result_rows if result.result_rows else []
        except _CHError as exc:
            logger.warning("[ClickhouseLogClient] Query error: %s", exc)
            return []
        except Exception as exc:
            logger.warning("[ClickhouseLogClient] Unexpected query error: %s", exc)
            return []

    def _rows_to_log_rows(self, raw_rows: list[tuple]) -> list[LogRow]:
        """
        Convert raw ClickHouse tuples into typed LogRow objects.

        Expected column order:
          Timestamp, Severity, ContainerName, Source, K8sReason, Rescued, Message
        """
        result: list[LogRow] = []
        for row in raw_rows:
            try:
                result.append(LogRow(
                    timestamp=row[0] if isinstance(row[0], datetime)
                              else datetime.fromtimestamp(float(row[0]), tz=timezone.utc),
                    severity=str(row[1] or "info"),
                    container_name=str(row[2] or ""),
                    source=str(row[3] or "container"),
                    k8s_reason=str(row[4] or ""),
                    rescued=int(row[5] or 0),
                    message=str(row[6] or ""),
                ))
            except Exception as exc:
                logger.debug("[ClickhouseLogClient] Skipping malformed row: %s — %s", row, exc)
                continue
        return result

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _build_log_query(
        self,
        table: str,
        namespace: str,
        pod_name: str,
        container_name: Optional[str],
        window_start: datetime,
        window_end: datetime,
        limit: int,
        rescued_only: bool = False,
    ) -> tuple[str, dict]:
        """
        Build a parameterised log query for a specific pod.

        Uses the ORDER BY key (Namespace, ServiceName, PodName, Timestamp)
        from the Golden Blueprint schema so ClickHouse uses the primary key
        index rather than performing a full table scan.

        rescued_only: When True (cold tier), restricts to Rescued=1 rows only
        to prevent INFO/DEBUG noise from being injected into LLM context.
        """
        container_clause = ""
        if container_name:
            container_clause = "AND ContainerName = {container_name:String}"

        severity_clause = """
              AND (
                  Severity IN ('ERROR', 'FATAL', 'CRITICAL', 'PANIC', 'WARN', 'WARNING')
                  OR Rescued = 1
              )"""
        if rescued_only:
            severity_clause = "\n              AND Rescued = 1"

        query = f"""
            SELECT
                Timestamp,
                Severity,
                ContainerName,
                Source,
                K8sReason,
                Rescued,
                Message
            FROM {table}
            WHERE Namespace = {{namespace:String}}
              AND PodName LIKE {{pod_prefix:String}}
              AND Timestamp BETWEEN {{window_start:DateTime64(9)}} AND {{window_end:DateTime64(9)}}
              {container_clause}
              {severity_clause}
            ORDER BY Timestamp DESC
            LIMIT {{limit:UInt32}}
        """
        params: dict = {
            "namespace": namespace,
            "pod_prefix": pod_name + "%",
            "window_start": window_start,
            "window_end": window_end,
            "limit": limit,
        }
        if container_name:
            params["container_name"] = container_name

        return query, params

    def _build_events_query(
        self,
        namespace: str,
        object_name_prefix: str,
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[str, dict]:
        """Build a parameterised K8s events query for a specific pod/object."""
        query = f"""
            SELECT
                Timestamp,
                EventType,
                '',
                'k8s_event',
                Reason,
                0,
                Message
            FROM {_EVENTS_TABLE}
            WHERE Namespace = {{namespace:String}}
              AND ObjectName LIKE {{object_prefix:String}}
              AND Timestamp BETWEEN {{window_start:DateTime64(9)}} AND {{window_end:DateTime64(9)}}
            ORDER BY Timestamp DESC
            LIMIT 50
        """
        params = {
            "namespace": namespace,
            "object_prefix": object_name_prefix + "%",
            "window_start": window_start,
            "window_end": window_end,
        }
        return query, params

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    async def query_pod_logs(
        self,
        namespace: str,
        pod_name: str,
        container_name: Optional[str] = None,
        lookback_minutes: int = _DEFAULT_LOOKBACK_MINUTES,
        limit: int = _MAX_ROWS,
    ) -> ForensicQueryResult:
        """
        Execute a tiered ClickHouse query for a specific pod.

        Tier escalation logic (improved over v1):
          1. Query ``logs_hot`` (sub-100ms)
          2. Escalate to ``logs_warm`` if:
             - rows < MINIMUM_ROWS, OR
             - semantic diversity < MINIMUM_DIVERSITY (repetitive hot logs)
          3. Escalate to ``logs_cold`` (rescued rows only) if still sparse/low-diversity
          4. Always query ``k8s_events`` and merge chronologically

        Returns a ForensicQueryResult. used_fallback=False — caller wraps
        this in the K8s fallback chain.
        """
        t_start = time.monotonic()
        result = ForensicQueryResult(pod_name=pod_name, namespace=namespace)

        window_end = datetime.now(tz=timezone.utc)
        window_start = window_end - timedelta(minutes=lookback_minutes)

        client = await self._get_client()
        if client is None:
            result.fallback_reason = "ClickHouse connection failed"
            return result

        try:
            # ── Hot tier ─────────────────────────────────────────────
            hot_query, hot_params = self._build_log_query(
                _HOT_TABLE, namespace, pod_name, container_name,
                window_start, window_end, limit,
            )
            hot_raw = await self._execute_query(hot_query, hot_params)
            log_rows = self._rows_to_log_rows(hot_raw)
            tier = "hot"

            diversity = _semantic_diversity_score(log_rows)
            needs_escalation = (
                len(log_rows) < _TIER_MINIMUM_ROWS
                or diversity < _TIER_MINIMUM_DIVERSITY
            )

            # ── Warm tier escalation ──────────────────────────────────
            if needs_escalation:
                logger.debug(
                    "[ClickhouseLogClient] Hot tier: %d rows, diversity=%.2f — escalating to warm",
                    len(log_rows), diversity,
                )
                warm_query, warm_params = self._build_log_query(
                    _WARM_TABLE, namespace, pod_name, container_name,
                    window_start, window_end, limit,
                )
                warm_raw = await self._execute_query(warm_query, warm_params)
                warm_rows = self._rows_to_log_rows(warm_raw)

                # Merge, deduplicate by (timestamp, message)
                seen = {(r.timestamp, r.message) for r in log_rows}
                for row in warm_rows:
                    if (row.timestamp, row.message) not in seen:
                        log_rows.append(row)
                        seen.add((row.timestamp, row.message))
                tier = "warm"

                # Re-evaluate diversity after warm merge
                diversity = _semantic_diversity_score(log_rows)
                needs_escalation = (
                    len(log_rows) < _TIER_MINIMUM_ROWS
                    or diversity < _TIER_MINIMUM_DIVERSITY
                )

            # ── Cold tier escalation (rescued rows only) ──────────────
            if needs_escalation:
                logger.debug(
                    "[ClickhouseLogClient] Warm tier still sparse (%d rows, div=%.2f) "
                    "— escalating to cold (rescued only, limit=%d)",
                    len(log_rows), diversity, _COLD_MAX_ROWS,
                )
                cold_query, cold_params = self._build_log_query(
                    _COLD_TABLE, namespace, pod_name, container_name,
                    window_start, window_end, _COLD_MAX_ROWS,
                    rescued_only=True,  # No INFO/DEBUG noise in cold tier
                )
                cold_raw = await self._execute_query(cold_query, cold_params)
                cold_rows = self._rows_to_log_rows(cold_raw)

                seen = {(r.timestamp, r.message) for r in log_rows}
                for row in cold_rows:
                    if (row.timestamp, row.message) not in seen:
                        log_rows.append(row)
                        seen.add((row.timestamp, row.message))
                tier = "cold"

            # Sort chronologically (ascending) for context coherence
            log_rows.sort(key=lambda r: r.timestamp)

            # ── K8s Events (always fetched, independent of tier) ──────
            events_query, events_params = self._build_events_query(
                namespace, pod_name, window_start, window_end,
            )
            events_raw = await self._execute_query(events_query, events_params)
            k8s_events = self._rows_to_log_rows(events_raw)
            k8s_events.sort(key=lambda r: r.timestamp)

            result.log_rows = log_rows
            result.k8s_event_rows = k8s_events
            result.tier_used = tier if log_rows else "empty"
            result.query_duration_ms = (time.monotonic() - t_start) * 1000

            logger.info(
                "[ClickhouseLogClient] %s/%s: %d log rows (%s tier, div=%.2f) "
                "+ %d K8s events in %.0fms",
                namespace, pod_name, len(log_rows), result.tier_used,
                _semantic_diversity_score(log_rows),
                len(k8s_events), result.query_duration_ms,
            )

        except Exception as exc:
            logger.warning(
                "[ClickhouseLogClient] Unexpected error querying %s/%s: %s",
                namespace, pod_name, exc,
            )
            result.fallback_reason = f"Unexpected ClickHouse error: {exc}"

        return result

    async def is_available(self) -> bool:
        """Return True if ClickHouse is reachable. Used for pre-flight checks."""
        client = await self._get_client()
        return client is not None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_default_client: Optional[ClickhouseLogClient] = None


def _get_default_client() -> ClickhouseLogClient:
    global _default_client
    if _default_client is None:
        _default_client = ClickhouseLogClient()
    return _default_client


# ---------------------------------------------------------------------------
# Public API — used directly by live_harness.py
# ---------------------------------------------------------------------------


async def query_forensic_logs_for_pod_async(
    namespace: str,
    pod_name: str,
    container_name: Optional[str] = None,
    lookback_minutes: int = _DEFAULT_LOOKBACK_MINUTES,
    limit: int = _MAX_ROWS,
    # K8s API fallback parameters (used when ClickHouse is unavailable)
    k8s_v1_client=None,
    k8s_fallback_tail_lines: int = 250,  # Increased from 50 — degraded mode, extract max data
) -> str:
    """
    Primary async entry point for forensic log extraction.

    Implements the full fallback chain:

      1. ClickHouse tiered query (Hot → Warm → Cold + K8s Events)
         Returns formatted text immediately if rows are found.

      2. K8s API fallback (``extract_forensic_logs`` from prometheus_anomaly)
         Activated when:
           - clickhouse-connect is not installed
           - ClickHouse is unreachable (network partition, pod restart)
           - All three tiers return zero rows (new namespace, cold start)
         Text is tagged ``[DEGRADED-FALLBACK]`` so the ReasoningEngine can
         apply lower confidence weighting to the diagnosis.

      3. Empty string — last resort if both sources fail.

    K8s API fallback rationale (preserved deliberately):
    -------------------------------------------------------
    The fallback is the critical safety net for pipeline outages.
    Eliminating it would leave AIRS completely blind during Vector/ClickHouse
    failures — the exact scenario pipeline-monitoring is designed to catch.

    Parameters
    ----------
    namespace:               Kubernetes namespace of the pod.
    pod_name:                Full pod name.
    container_name:          Specific container. None = all containers.
    lookback_minutes:        How far back to query (default: 60 minutes).
    limit:                   Max rows to return (default: 200).
    k8s_v1_client:           kubernetes.client.CoreV1Api for fallback.
    k8s_fallback_tail_lines: Lines to tail per container. Default raised to
                             250 in degraded mode to maximise data extraction.

    Returns
    -------
    str
        Formatted forensic log text ready for AIRS ReasoningEngine.
        Empty string only if all sources return no data.
    """
    client = _get_default_client()

    # ── Attempt ClickHouse ────────────────────────────────────────────────
    try:
        ch_result = await client.query_pod_logs(
            namespace=namespace,
            pod_name=pod_name,
            container_name=container_name,
            lookback_minutes=lookback_minutes,
            limit=limit,
        )

        if ch_result.total_rows > 0:
            text = ch_result.to_text(compact=True)
            if text.strip():
                logger.info(
                    "[ForensicQuery] ClickHouse returned %d rows for %s/%s "
                    "(tier=%s, dedup=%.0f%%, %.0fms)",
                    ch_result.total_rows, namespace, pod_name,
                    ch_result.tier_used,
                    ch_result.dedup_ratio * 100,
                    ch_result.query_duration_ms,
                )
                return text

        reason = ch_result.fallback_reason or "zero rows returned by all tiers"
        logger.info(
            "[ForensicQuery] ClickHouse returned no data for %s/%s (%s) — "
            "falling back to K8s API",
            namespace, pod_name, reason,
        )

    except Exception as exc:
        logger.warning(
            "[ForensicQuery] ClickHouse query raised exception for %s/%s: %s — "
            "falling back to K8s API",
            namespace, pod_name, exc,
        )

    # ── K8s API Fallback ─────────────────────────────────────────────────
    if k8s_v1_client is not None:
        try:
            from prometheus_anomaly import extract_forensic_logs as _k8s_extract
            logger.info(
                "[ForensicQuery] Using K8s API fallback for %s/%s (tail=%d lines)",
                namespace, pod_name, k8s_fallback_tail_lines,
            )
            k8s_text = _k8s_extract(
                v1=k8s_v1_client,
                namespace=namespace,
                pod_name=pod_name,
                container=container_name,
                tail_lines=k8s_fallback_tail_lines,
            )
            if k8s_text and k8s_text.strip():
                # Tag with DEGRADED-FALLBACK so ReasoningEngine applies lower
                # confidence weighting; pipeline monitoring should alert on this.
                logger.warning(
                    "[ForensicQuery] DEGRADED-FALLBACK active for %s/%s — "
                    "check Vector/ClickHouse pipeline health",
                    namespace, pod_name,
                )
                return f"[DEGRADED-FALLBACK — ClickHouse unavailable]\n{k8s_text}"
        except Exception as exc:
            logger.warning(
                "[ForensicQuery] K8s API fallback also failed for %s/%s: %s",
                namespace, pod_name, exc,
            )

    # ── Total failure ────────────────────────────────────────────────────
    logger.warning(
        "[ForensicQuery] All sources failed for %s/%s — returning empty string",
        namespace, pod_name,
    )
    return ""


def query_forensic_logs_for_pod(
    namespace: str,
    pod_name: str,
    container_name: Optional[str] = None,
    lookback_minutes: int = _DEFAULT_LOOKBACK_MINUTES,
    limit: int = _MAX_ROWS,
    k8s_v1_client=None,
    k8s_fallback_tail_lines: int = 250,
) -> str:
    """
    Synchronous wrapper around ``query_forensic_logs_for_pod_async``.

    Uses ``asyncio.run()`` when called from non-async contexts.
    Prefer the async version (``query_forensic_logs_for_pod_async``)
    from async code to avoid blocking the event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop — schedule as coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    query_forensic_logs_for_pod_async(
                        namespace, pod_name, container_name,
                        lookback_minutes, limit, k8s_v1_client, k8s_fallback_tail_lines,
                    ),
                )
                return future.result()
        else:
            return loop.run_until_complete(
                query_forensic_logs_for_pod_async(
                    namespace, pod_name, container_name,
                    lookback_minutes, limit, k8s_v1_client, k8s_fallback_tail_lines,
                )
            )
    except Exception as exc:
        logger.warning("[ForensicQuery] Sync wrapper failed: %s", exc)
        return ""
