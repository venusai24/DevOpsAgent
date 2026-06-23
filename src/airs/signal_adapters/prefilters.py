"""
Deterministic Pre-Filters — Module 1.6.

Pure mathematical/statistical functions used by SignalAdapters to reduce
raw MCP responses before they are passed to the LLM. No LLM involved here.

Each pre-filter:
  - Is idempotent (same input → same output)
  - Uses only deterministic math (Z-score, regex, sorting, dedup)
  - Returns a filtered subset with filter_stats metadata
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any


def zscore_filter(
    data_points: list[dict],
    value_key: str,
    z_threshold: float = 2.0,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Filter a list of metric data points by Z-score.

    Retains only points where |Z| > threshold (anomalous points).
    If all points are within threshold (no anomaly), returns the top-3
    highest absolute-Z points to preserve signal.

    Args:
        data_points:   List of dicts, each with a numeric `value_key`.
        value_key:     Key in each dict holding the numeric value.
        z_threshold:   Z-score threshold for anomaly detection (default 2σ).

    Returns:
        (filtered_points, stats) — filtered subset and stats about the filter.
    """
    if not data_points:
        return [], {"input_count": 0, "output_count": 0, "method": "zscore"}

    values = []
    for dp in data_points:
        try:
            values.append(float(dp[value_key]))
        except (KeyError, TypeError, ValueError):
            values.append(0.0)

    n = len(values)
    if n == 0:
        return [], {"input_count": 0, "output_count": 0, "method": "zscore"}

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0

    z_scores = [(v - mean) / std for v in values]

    anomalous = [
        (i, dp, z_scores[i])
        for i, dp in enumerate(data_points)
        if abs(z_scores[i]) > z_threshold
    ]

    if not anomalous:
        # No anomaly found — return top-3 by absolute Z-score anyway
        ranked = sorted(
            enumerate(data_points),
            key=lambda x: abs(z_scores[x[0]]),
            reverse=True,
        )
        filtered = [dp for _, dp in ranked[:3]]
    else:
        # Sort by descending |Z-score|
        anomalous.sort(key=lambda x: abs(x[2]), reverse=True)
        filtered = [dp for _, dp, _ in anomalous]

    stats = {
        "method": "zscore",
        "z_threshold": z_threshold,
        "input_count": n,
        "output_count": len(filtered),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "max_z": round(max(abs(z) for z in z_scores), 4),
        "anomaly_found": bool(anomalous),
    }
    return filtered, stats


def dedup_log_patterns(
    log_lines: list[str],
    max_patterns: int = 20,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Deduplicate log lines by pattern (variable parts replaced with placeholders).

    Strategy:
    1. Replace variable parts (timestamps, UUIDs, IPs, numbers) with <VAR>
    2. Group by normalised pattern
    3. Keep one representative per pattern + count
    4. Return top `max_patterns` by count

    Args:
        log_lines:    Raw log lines as strings.
        max_patterns: Maximum number of distinct patterns to return.

    Returns:
        (pattern_records, stats) where each record has {pattern, example, count}.
    """
    if not log_lines:
        return [], {"input_count": 0, "output_count": 0, "method": "log_dedup"}

    # Pattern normalisation: replace variables
    _var_patterns = [
        (re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I), '<UUID>'),
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), '<IP>'),
        (re.compile(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.\d]*Z?\b'), '<TS>'),
        (re.compile(r'\b[0-9a-f]{24,64}\b', re.I), '<HEX>'),
        (re.compile(r'\b\d{5,}\b'), '<NUM>'),
    ]

    pattern_groups: dict[str, dict] = {}

    for line in log_lines:
        normalised = line
        for regex, placeholder in _var_patterns:
            normalised = regex.sub(placeholder, normalised)

        # Further normalise: collapse multiple spaces
        normalised = re.sub(r'\s+', ' ', normalised).strip()

        if normalised not in pattern_groups:
            pattern_groups[normalised] = {"pattern": normalised, "example": line, "count": 0}
        pattern_groups[normalised]["count"] += 1

    # Sort by count descending, take top max_patterns
    sorted_patterns = sorted(
        pattern_groups.values(), key=lambda x: x["count"], reverse=True
    )[:max_patterns]

    stats = {
        "method": "log_dedup",
        "input_count": len(log_lines),
        "output_count": len(sorted_patterns),
        "unique_patterns": len(pattern_groups),
        "dedup_ratio": round(1 - len(sorted_patterns) / max(len(log_lines), 1), 3),
    }
    return sorted_patterns, stats


def filter_error_spans(
    spans: list[dict],
    error_tag_key: str = "error",
    status_code_key: str = "status_code",
    max_spans: int = 20,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Filter distributed trace spans to retain only error + critical path spans.

    Strategy:
    1. Retain spans with error=true or http.status_code >= 500
    2. If no error spans, retain the slowest N spans
    3. Cap at max_spans

    Args:
        spans:           List of span dicts from Jaeger/Tempo.
        error_tag_key:   Tag key indicating an error (bool or 'true'/'false').
        status_code_key: Tag key for HTTP status code.
        max_spans:       Maximum spans to return.

    Returns:
        (filtered_spans, stats).
    """
    if not spans:
        return [], {"input_count": 0, "output_count": 0, "method": "error_span_filter"}

    def is_error(span: dict) -> bool:
        tags = span.get("tags", {})
        # Handle both dict and list-of-kv formats
        if isinstance(tags, dict):
            err = tags.get(error_tag_key, False)
            sc = tags.get(status_code_key, 0)
        elif isinstance(tags, list):
            kv = {t.get("key"): t.get("value") for t in tags if isinstance(t, dict)}
            err = kv.get(error_tag_key, False)
            sc = kv.get(status_code_key, 0)
        else:
            return False

        if isinstance(err, str):
            err = err.lower() == "true"
        try:
            sc = int(sc)
        except (TypeError, ValueError):
            sc = 0

        return bool(err) or sc >= 500

    error_spans = [s for s in spans if is_error(s)]

    if not error_spans:
        # No error spans — return slowest spans
        def duration(s: dict) -> float:
            try:
                return float(s.get("duration", 0))
            except (TypeError, ValueError):
                return 0.0
        error_spans = sorted(spans, key=duration, reverse=True)[:max_spans]
        method = "slowest_spans"
    else:
        error_spans = error_spans[:max_spans]
        method = "error_span_filter"

    stats = {
        "method": method,
        "input_count": len(spans),
        "output_count": len(error_spans),
        "error_span_count": sum(1 for s in spans if is_error(s)),
    }
    return error_spans, stats


def filter_k8s_events(
    events: list[dict],
    namespace: Optional[str] = None,
    reason_blocklist: Optional[list[str]] = None,
    max_events: int = 30,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Filter Kubernetes events to retain anomalous/warning events.

    Strategy:
    1. Filter by namespace if specified
    2. Exclude benign reasons (Scheduled, Pulling, Pulled, Started...)
    3. Prioritise Warning type events over Normal
    4. Cap at max_events

    Args:
        events:          List of K8s event dicts.
        namespace:       If set, filter to this namespace only.
        reason_blocklist: Reasons to exclude (benign events).
        max_events:      Maximum events to return.

    Returns:
        (filtered_events, stats).
    """
    from typing import Optional

    if not events:
        return [], {"input_count": 0, "output_count": 0, "method": "k8s_event_filter"}

    if reason_blocklist is None:
        reason_blocklist = [
            "Scheduled", "Pulled", "Pulling", "Started", "Created",
            "SuccessfulCreate", "Sync", "LeaderElection",
        ]

    filtered = events

    if namespace:
        filtered = [
            e for e in filtered
            if e.get("metadata", {}).get("namespace") == namespace
            or e.get("namespace") == namespace
        ]

    # Exclude benign reasons
    filtered = [
        e for e in filtered
        if e.get("reason") not in reason_blocklist
    ]

    # Sort: Warning first, then by count descending
    def sort_key(e: dict) -> tuple:
        event_type = e.get("type", "Normal")
        count = int(e.get("count", 1))
        return (0 if event_type == "Warning" else 1, -count)

    filtered.sort(key=sort_key)
    filtered = filtered[:max_events]

    stats = {
        "method": "k8s_event_filter",
        "input_count": len(events),
        "output_count": len(filtered),
        "namespace_filter": namespace,
    }
    return filtered, stats


# Allow Optional usage in filter_k8s_events without top-level import
from typing import Optional  # noqa: E402
