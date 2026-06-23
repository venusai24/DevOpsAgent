"""
CSV Analytics — Deterministic Analytical Algorithms.

All computations in this module are purely deterministic and mathematical.
No LLM is involved. These functions produce the verified, factual insights
that are later fed into the LLM reasoning layer as evidence.

Functions:
  - detect_metric_anomalies(): Z-score over metric time series via DuckDB SQL
  - extract_log_patterns(): Deduplicate log messages, count unique patterns
  - find_slow_spans(): Extract high-duration trace spans (>P95 threshold)
  - compute_correlation(): Cross-table Pearson correlation via DuckDB
  - summarize_time_window(): Rolling time-window aggregation
"""
from __future__ import annotations

import logging
import math
from typing import Any

from airs.csv_engine.engine import CsvEngine

log = logging.getLogger(__name__)

# Z-score threshold for anomaly detection on metric values.
# Values more than N standard deviations from the mean are flagged.
ZSCORE_THRESHOLD = 2.5

# Duration threshold multiplier for trace span analysis.
# Spans with duration > P95 * TRACE_SLOW_MULTIPLIER are flagged.
TRACE_SLOW_MULTIPLIER = 2.0


def detect_metric_anomalies(
    engine: CsvEngine,
    table: str = "prometheus_metrics",
    kpi_filter: str | None = None,
    cmdb_filter: str | None = None,
) -> dict[str, Any]:
    """
    Detect anomalous metric values using Z-score, computed entirely in DuckDB.

    DuckDB computes the mean and standard deviation per kpi_name+cmdb_id
    combination using a single SQL pass — no data is pulled into Python.
    Only the anomalous rows (Z-score > threshold) are returned.

    Args:
        engine:       Initialized CsvEngine.
        table:        DuckDB view name (default: prometheus_metrics).
        kpi_filter:   Optional KPI name to restrict analysis (e.g. 'cpu_usage_percent').
        cmdb_filter:  Optional CMDB ID to restrict analysis (e.g. 'srv-app-01').

    Returns:
        dict with:
          - anomalies: list of dicts [kpi_name, cmdb_id, timestamp, value, z_score]
          - kpis_analyzed: how many unique KPIs were scanned
          - error: None on success
    """
    where_clauses = []
    if kpi_filter:
        where_clauses.append(f"kpi_name = '{kpi_filter}'")
    if cmdb_filter:
        where_clauses.append(f"cmdb_id = '{cmdb_filter}'")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Single-pass Z-score via window functions — runs entirely inside DuckDB
    sql = f"""
    WITH stats AS (
        SELECT
            kpi_name,
            cmdb_id,
            timestamp,
            value,
            AVG(value) OVER (PARTITION BY kpi_name, cmdb_id) AS mean_val,
            STDDEV_POP(value) OVER (PARTITION BY kpi_name, cmdb_id) AS std_val
        FROM {table}
        {where_sql}
    )
    SELECT
        kpi_name,
        cmdb_id,
        timestamp,
        ROUND(value, 4) AS value,
        ROUND(
            CASE
                WHEN std_val = 0 OR std_val IS NULL THEN 0
                ELSE ABS(value - mean_val) / std_val
            END, 3
        ) AS z_score
    FROM stats
    WHERE
        std_val > 0
        AND ABS(value - mean_val) / std_val > {ZSCORE_THRESHOLD}
    ORDER BY z_score DESC
    """

    result = engine.query(sql)
    if result["error"]:
        return {"anomalies": [], "kpis_analyzed": 0, "error": result["error"]}

    # Count unique KPIs for context
    kpi_count_result = engine.query(
        f"SELECT COUNT(DISTINCT kpi_name) AS cnt FROM {table} {where_sql}"
    )
    kpis_analyzed = (
        kpi_count_result["rows"][0]["cnt"]
        if kpi_count_result["rows"] else 0
    )

    return {
        "anomalies": result["rows"],
        "anomaly_count": len(result["rows"]),
        "kpis_analyzed": kpis_analyzed,
        "zscore_threshold": ZSCORE_THRESHOLD,
        "truncated": result["truncated"],
        "error": None,
    }


def extract_log_patterns(
    engine: CsvEngine,
    table: str = "loki_logs",
    min_count: int = 2,
    severity_filter: str | None = None,
) -> dict[str, Any]:
    """
    Deterministically deduplicate log messages and count unique patterns.

    DuckDB extracts the first 120 characters of each log message as a
    "pattern key" and groups identical or near-identical messages together.
    This collapses thousands of repeated error lines into a compact
    summary (e.g., "Connection refused to postgres:5432 — 1,247 occurrences").

    Args:
        engine:          Initialized CsvEngine.
        table:           DuckDB view (default: loki_logs).
        min_count:       Only return patterns that appear >= min_count times.
        severity_filter: Optional prefix to filter (e.g., 'ERROR', 'WARN').

    Returns:
        dict with:
          - patterns: list of dicts [pattern, cmdb_id, log_name, count, first_seen, last_seen]
          - total_log_lines: total raw line count in the table
          - unique_patterns: number of unique patterns found
    """
    severity_clause = ""
    if severity_filter:
        severity_clause = f"AND value LIKE '{severity_filter}%'"

    # Use SUBSTR to create a pattern key from the first 120 chars
    sql = f"""
    SELECT
        SUBSTR(TRIM(value), 1, 120) AS pattern,
        cmdb_id,
        log_name,
        COUNT(*) AS occurrence_count,
        MIN(timestamp) AS first_seen,
        MAX(timestamp) AS last_seen
    FROM {table}
    WHERE value IS NOT NULL {severity_clause}
    GROUP BY pattern, cmdb_id, log_name
    HAVING COUNT(*) >= {min_count}
    ORDER BY occurrence_count DESC
    """

    result = engine.query(sql)

    # Total raw line count — cheap COUNT(*)
    count_result = engine.query(f"SELECT COUNT(*) AS cnt FROM {table}")
    total = count_result["rows"][0]["cnt"] if count_result["rows"] else 0

    return {
        "patterns": result["rows"],
        "unique_patterns": len(result["rows"]),
        "total_log_lines": total,
        "truncated": result["truncated"],
        "error": result["error"],
    }


def find_slow_spans(
    engine: CsvEngine,
    table: str = "jaeger_traces",
    cmdb_filter: str | None = None,
    percentile_threshold: float = 95.0,
) -> dict[str, Any]:
    """
    Identify abnormally slow trace spans.

    Computes the P95 duration per service (cmdb_id) entirely in DuckDB,
    then returns only spans exceeding 2× the P95 threshold.

    Args:
        engine:               Initialized CsvEngine.
        table:                DuckDB view (default: jaeger_traces).
        cmdb_filter:          Optional service name to restrict analysis.
        percentile_threshold: Percentile baseline (default P95).

    Returns:
        dict with:
          - slow_spans: list of dicts [timestamp, cmdb_id, span_id, trace_id, duration, p95_baseline]
          - services_analyzed: unique service count
    """
    where_sql = f"WHERE cmdb_id = '{cmdb_filter}'" if cmdb_filter else ""
    pct_fraction = percentile_threshold / 100.0

    sql = f"""
    WITH baselines AS (
        SELECT
            cmdb_id,
            QUANTILE_CONT(duration, {pct_fraction}) AS p_threshold
        FROM {table}
        {where_sql}
        GROUP BY cmdb_id
    )
    SELECT
        t.timestamp,
        t.cmdb_id,
        t.span_id,
        t.trace_id,
        ROUND(t.duration, 2) AS duration_us,
        ROUND(b.p_threshold, 2) AS p{int(percentile_threshold)}_baseline_us
    FROM {table} t
    JOIN baselines b ON t.cmdb_id = b.cmdb_id
    WHERE t.duration > b.p_threshold * {TRACE_SLOW_MULTIPLIER}
    ORDER BY t.duration DESC
    """

    result = engine.query(sql)

    svc_result = engine.query(
        f"SELECT COUNT(DISTINCT cmdb_id) AS cnt FROM {table} {where_sql}"
    )
    services_analyzed = svc_result["rows"][0]["cnt"] if svc_result["rows"] else 0

    return {
        "slow_spans": result["rows"],
        "slow_span_count": len(result["rows"]),
        "services_analyzed": services_analyzed,
        "percentile_threshold": percentile_threshold,
        "slow_multiplier": TRACE_SLOW_MULTIPLIER,
        "truncated": result["truncated"],
        "error": result["error"],
    }


def compute_time_window_summary(
    engine: CsvEngine,
    table: str,
    time_col: str,
    value_col: str,
    window_minutes: int = 5,
    kpi_filter: str | None = None,
) -> dict[str, Any]:
    """
    Produce a rolling time-bucket aggregation summary.

    Groups records into fixed-width time buckets and computes
    min/max/avg/count per bucket. This gives the LLM a compact
    temporal overview (e.g., how error rate changed over 5-min windows)
    without injecting individual data points.

    Args:
        engine:         Initialized CsvEngine.
        table:          DuckDB view name.
        time_col:       Column name holding the Unix timestamp.
        value_col:      Column name to aggregate.
        window_minutes: Width of each time bucket in minutes.
        kpi_filter:     Optional: restrict to one KPI name.

    Returns:
        dict with:
          - buckets: list of [bucket_start, count, min_val, max_val, avg_val]
    """
    window_seconds = window_minutes * 60
    kpi_clause = f"AND kpi_name = '{kpi_filter}'" if kpi_filter else ""

    sql = f"""
    SELECT
        EPOCH(DATE_TRUNC('minute',
            TO_TIMESTAMP(CAST({time_col} AS DOUBLE))
        ))
        / {window_seconds} * {window_seconds} AS bucket_start,
        COUNT(*) AS data_points,
        ROUND(MIN(CAST({value_col} AS DOUBLE)), 4) AS min_val,
        ROUND(MAX(CAST({value_col} AS DOUBLE)), 4) AS max_val,
        ROUND(AVG(CAST({value_col} AS DOUBLE)), 4) AS avg_val
    FROM {table}
    WHERE {value_col} IS NOT NULL {kpi_clause}
    GROUP BY bucket_start
    ORDER BY bucket_start ASC
    """

    result = engine.query(sql)
    return {
        "buckets": result["rows"],
        "bucket_count": len(result["rows"]),
        "window_minutes": window_minutes,
        "truncated": result["truncated"],
        "error": result["error"],
    }


def compute_cross_table_correlation(
    engine: CsvEngine,
    metric_kpi: str,
    log_pattern_prefix: str,
    time_col_metrics: str = "timestamp",
    time_col_logs: str = "timestamp",
    window_minutes: int = 5,
) -> dict[str, Any]:
    """
    Compute Pearson correlation between a metric KPI and log error rate.

    Joins prometheus_metrics and loki_logs on 5-minute time buckets,
    then computes the Pearson correlation coefficient between the
    metric value and the log occurrence count in that bucket.

    This is purely deterministic math inside DuckDB.

    Returns:
        dict with:
          - correlation: Pearson r in [-1, 1]
          - interpretation: human-readable correlation strength
          - sample_points: number of time buckets used
    """
    window_seconds = window_minutes * 60

    sql = f"""
    WITH metric_buckets AS (
        SELECT
            CAST({time_col_metrics} AS DOUBLE) / {window_seconds}
                * {window_seconds} AS bucket,
            AVG(CAST(value AS DOUBLE)) AS avg_metric
        FROM prometheus_metrics
        WHERE kpi_name = '{metric_kpi}'
        GROUP BY bucket
    ),
    log_buckets AS (
        SELECT
            CAST({time_col_logs} AS DOUBLE) / {window_seconds}
                * {window_seconds} AS bucket,
            COUNT(*) AS error_count
        FROM loki_logs
        WHERE value LIKE '{log_pattern_prefix}%'
        GROUP BY bucket
    ),
    joined AS (
        SELECT m.avg_metric, l.error_count
        FROM metric_buckets m
        JOIN log_buckets l ON m.bucket = l.bucket
    )
    SELECT
        CORR(avg_metric, error_count) AS pearson_r,
        COUNT(*) AS sample_points
    FROM joined
    """

    result = engine.query(sql)
    if result["error"] or not result["rows"]:
        return {"correlation": None, "sample_points": 0, "error": result["error"]}

    row = result["rows"][0]
    r = row.get("pearson_r")

    interpretation = _interpret_correlation(r)

    return {
        "correlation": round(r, 4) if r is not None else None,
        "interpretation": interpretation,
        "sample_points": row.get("sample_points", 0),
        "window_minutes": window_minutes,
        "error": None,
    }


def _interpret_correlation(r: float | None) -> str:
    if r is None:
        return "Insufficient data to compute correlation."
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r >= 0.8:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "negligible"
    return f"{strength.capitalize()} {direction} correlation (r={r:.3f})"
