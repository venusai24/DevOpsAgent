"""
CSV Intelligence Tools — Phase B.

These are the deterministic tools the LLM calls when it needs to reason
over CSV telemetry data. They form the primary interface between the
agent's decision-making layer and the raw CSV files.

Design philosophy:
  - Tools are pure functions: (engine, args) → JSON-serializable dict
  - No raw CSV rows reach the LLM context. Every tool returns
    pre-computed insights (aggregations, anomaly lists, pattern counts).
  - Row limits are enforced inside the engine — not here.
  - Tools document their own output schemas so the LLM can reason
    about what to call next.

Available Tools (for MCP registry):
  1. csv_schema_discovery     → What CSVs exist, columns, row counts
  2. csv_sql_query            → Run arbitrary SQL (with row limit guard)
  3. csv_anomaly_detection    → Z-score metric anomalies
  4. csv_log_pattern_extract  → Deduplicated log patterns + counts
  5. csv_slow_span_detection  → High-duration trace spans
  6. csv_time_window_summary  → Rolling time-bucket aggregation
  7. csv_correlation_analysis → Pearson correlation across two signals
"""
from __future__ import annotations

import logging
from typing import Any

from airs.csv_engine.analytics import (
    compute_cross_table_correlation,
    compute_time_window_summary,
    detect_metric_anomalies,
    extract_log_patterns,
    find_slow_spans,
)
from airs.csv_engine.engine import CsvEngine

log = logging.getLogger(__name__)


def csv_schema_discovery(engine: CsvEngine) -> dict[str, Any]:
    """
    Tool 1: Discover what CSV data is available for this incident.

    Returns each registered table's name, source file, column names
    with inferred data types, and total row count.

    The agent MUST call this first before any other CSV tool to understand
    what data is available and what columns it can query.

    Output schema:
        {
            "registered_tables": {
                "<table_name>": {
                    "source_file": str,
                    "row_count": int,
                    "columns": [{"name": str, "type": str}, ...]
                }
            },
            "missing_files": [str],
            "available_table_names": [str]
        }
    """
    schema = engine.initialize()
    schema["available_table_names"] = list(schema["registered_tables"].keys())
    return schema


def csv_sql_query(engine: CsvEngine, query: str) -> dict[str, Any]:
    """
    Tool 2: Execute a deterministic SQL query against the CSV data.

    Use this for targeted filtering, projection, grouping, and aggregation.
    Results are hard-capped at 200 rows. If your result would exceed 200 rows,
    you MUST refine your query using:
      - WHERE clauses to filter rows
      - GROUP BY + COUNT/AVG/SUM to aggregate
      - LIMIT with ORDER BY to select top-N

    Available table names: prometheus_metrics, loki_logs, opensearch_logs,
                           jaeger_traces, tempo_traces

    Example queries:
      SELECT kpi_name, COUNT(*), AVG(value) FROM prometheus_metrics GROUP BY kpi_name
      SELECT cmdb_id, COUNT(*) AS errors FROM loki_logs WHERE value LIKE 'ERROR%' GROUP BY cmdb_id
      SELECT * FROM jaeger_traces WHERE cmdb_id = 'payment-service' ORDER BY duration DESC LIMIT 10

    Output schema:
        {
            "columns": [str],
            "rows": [{column: value}],
            "row_count": int,
            "truncated": bool,
            "truncation_message": str | None,
            "error": str | None
        }
    """
    return engine.query(query)


def csv_anomaly_detection(
    engine: CsvEngine,
    kpi_filter: str | None = None,
    cmdb_filter: str | None = None,
) -> dict[str, Any]:
    """
    Tool 3: Detect anomalous metric values using Z-score analysis.

    Computes Z-scores for all metric readings in prometheus_metrics.
    Returns only the data points that are statistically anomalous
    (|Z-score| > 2.5σ from the per-KPI mean).

    This tool is the CORRECT way to identify metric spikes and dips.
    Never ask the LLM to reason about raw metric rows.

    Args:
        kpi_filter:  Optional KPI name to restrict analysis (e.g., 'cpu_usage_percent').
        cmdb_filter: Optional instance/host to restrict analysis (e.g., 'srv-app-01').

    Output schema:
        {
            "anomalies": [{"kpi_name", "cmdb_id", "timestamp", "value", "z_score"}],
            "anomaly_count": int,
            "kpis_analyzed": int,
            "zscore_threshold": float,
            "truncated": bool
        }
    """
    return detect_metric_anomalies(engine, kpi_filter=kpi_filter, cmdb_filter=cmdb_filter)


def csv_log_pattern_extract(
    engine: CsvEngine,
    table: str = "loki_logs",
    severity_filter: str | None = None,
    min_count: int = 2,
) -> dict[str, Any]:
    """
    Tool 4: Extract and deduplicate log patterns with occurrence counts.

    Collapses thousands of repeated log lines into a compact list of
    unique patterns, each with a count and time range. This is the
    correct tool for log analysis — do not query raw log rows.

    Args:
        table:           'loki_logs' or 'opensearch_logs'
        severity_filter: Optional prefix filter: 'ERROR', 'WARN', 'INFO'
        min_count:       Minimum occurrences to include a pattern

    Output schema:
        {
            "patterns": [{"pattern", "cmdb_id", "log_name", "occurrence_count", "first_seen", "last_seen"}],
            "unique_patterns": int,
            "total_log_lines": int
        }
    """
    return extract_log_patterns(
        engine, table=table, severity_filter=severity_filter, min_count=min_count
    )


def csv_slow_span_detection(
    engine: CsvEngine,
    table: str = "jaeger_traces",
    cmdb_filter: str | None = None,
    percentile_threshold: float = 95.0,
) -> dict[str, Any]:
    """
    Tool 5: Find abnormally slow trace spans.

    Computes the P95 duration per service and returns spans
    exceeding 2× the P95 baseline. Use this for latency root cause
    analysis — do not query raw trace rows.

    Args:
        table:                'jaeger_traces' or 'tempo_traces'
        cmdb_filter:          Optional service name (e.g., 'payment-service')
        percentile_threshold: Percentile baseline, default P95

    Output schema:
        {
            "slow_spans": [{"timestamp", "cmdb_id", "span_id", "trace_id", "duration_us", "p95_baseline_us"}],
            "slow_span_count": int,
            "services_analyzed": int
        }
    """
    return find_slow_spans(
        engine, table=table, cmdb_filter=cmdb_filter, percentile_threshold=percentile_threshold
    )


def csv_time_window_summary(
    engine: CsvEngine,
    table: str,
    time_col: str,
    value_col: str,
    window_minutes: int = 5,
    kpi_filter: str | None = None,
) -> dict[str, Any]:
    """
    Tool 6: Summarize a metric/log column in rolling time buckets.

    Groups data into fixed time windows and returns
    min/max/avg/count per window. Use this to understand trends
    and rate-of-change over time without injecting individual data points.

    Args:
        table:          Table name (e.g., 'prometheus_metrics')
        time_col:       Column holding Unix timestamps (e.g., 'timestamp')
        value_col:      Column to aggregate (e.g., 'value')
        window_minutes: Time bucket width in minutes (default 5)
        kpi_filter:     Optional KPI name for metric tables

    Output schema:
        {
            "buckets": [{"bucket_start", "data_points", "min_val", "max_val", "avg_val"}],
            "bucket_count": int,
            "window_minutes": int
        }
    """
    return compute_time_window_summary(
        engine,
        table=table,
        time_col=time_col,
        value_col=value_col,
        window_minutes=window_minutes,
        kpi_filter=kpi_filter,
    )


def csv_correlation_analysis(
    engine: CsvEngine,
    metric_kpi: str,
    log_pattern_prefix: str,
    window_minutes: int = 5,
) -> dict[str, Any]:
    """
    Tool 7: Compute Pearson correlation between a metric KPI and log error rate.

    Joins prometheus_metrics and loki_logs on time buckets and computes
    correlation between metric values and log occurrence counts.

    Use this to confirm or refute a causal hypothesis, e.g.:
    "Does high cpu_usage correlate with increased ERROR log frequency?"

    Args:
        metric_kpi:          KPI name in prometheus_metrics (e.g., 'cpu_usage_percent')
        log_pattern_prefix:  Log message prefix to count (e.g., 'ERROR: Connection refused')
        window_minutes:      Time bucket size for joining the two signals

    Output schema:
        {
            "correlation": float | None,    # Pearson r in [-1, 1]
            "interpretation": str,          # human-readable strength/direction
            "sample_points": int            # number of time buckets compared
        }
    """
    return compute_cross_table_correlation(
        engine,
        metric_kpi=metric_kpi,
        log_pattern_prefix=log_pattern_prefix,
        window_minutes=window_minutes,
    )


# Tool registry map: tool_name → (function, description_for_LLM)
CSV_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "csv_schema_discovery": {
        "fn": csv_schema_discovery,
        "description": "Discover available CSV tables, their columns, and row counts. Call this first.",
        "args": [],
    },
    "csv_sql_query": {
        "fn": csv_sql_query,
        "description": "Run SQL against available CSV tables. Results capped at 200 rows. Use aggregations.",
        "args": ["query"],
    },
    "csv_anomaly_detection": {
        "fn": csv_anomaly_detection,
        "description": "Z-score anomaly detection on prometheus_metrics. Returns only anomalous data points.",
        "args": ["kpi_filter?", "cmdb_filter?"],
    },
    "csv_log_pattern_extract": {
        "fn": csv_log_pattern_extract,
        "description": "Deduplicate log lines into unique patterns with counts. Use instead of raw log queries.",
        "args": ["table?", "severity_filter?", "min_count?"],
    },
    "csv_slow_span_detection": {
        "fn": csv_slow_span_detection,
        "description": "Identify abnormally slow trace spans (>2× P95). Use for latency root cause analysis.",
        "args": ["table?", "cmdb_filter?", "percentile_threshold?"],
    },
    "csv_time_window_summary": {
        "fn": csv_time_window_summary,
        "description": "Aggregate a metric/log column into rolling time buckets (min/max/avg/count per window).",
        "args": ["table", "time_col", "value_col", "window_minutes?", "kpi_filter?"],
    },
    "csv_correlation_analysis": {
        "fn": csv_correlation_analysis,
        "description": "Pearson correlation between a metric KPI and log error rate across time windows.",
        "args": ["metric_kpi", "log_pattern_prefix", "window_minutes?"],
    },
}
