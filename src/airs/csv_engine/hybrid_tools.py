"""
Hybrid Intelligence Tools — Phase B.

These tools combine deterministic data pipelines (DuckDB) with
pre-packaged analysis logic, completely isolating the LLM from
raw data arrays or the need to write raw SQL.

The main orchestration LLM only receives high-level summaries
from these tools, drastically reducing prompt size and cognitive load.
"""
from __future__ import annotations

import logging
from typing import Any

from airs.csv_engine.engine import CsvEngine
from airs.csv_engine.tools import csv_anomaly_detection, csv_log_pattern_extract

log = logging.getLogger(__name__)


def csv_analyze_metric_health(
    engine: CsvEngine,
    metric_kpi: str,
    cmdb_filter: str | None = None,
) -> dict[str, Any]:
    """
    Hybrid Tool: Analyzes a specific metric for health.
    
    Instead of returning an array of anomalies, it returns a compressed
    natural language summary indicating if the metric is healthy or not.
    """
    raw_result = csv_anomaly_detection(engine, kpi_filter=metric_kpi, cmdb_filter=cmdb_filter)
    
    count = raw_result.get("anomaly_count", 0)
    if count == 0:
        return {
            "status": "HEALTHY",
            "summary": f"Metric '{metric_kpi}' is perfectly healthy. 0 anomalies detected in the current window. Prune hypotheses related to {metric_kpi} degradation."
        }
    
    anomalies = raw_result.get("anomalies", [])
    max_z = max([abs(a.get("z_score", 0)) for a in anomalies]) if anomalies else 0
    
    return {
        "status": "ANOMALOUS",
        "summary": f"Metric '{metric_kpi}' is highly anomalous! Detected {count} outlier points with a maximum Z-score of {max_z:.2f}σ. This is a critical causal indicator."
    }


def csv_extract_log_errors(
    engine: CsvEngine,
    cmdb_filter: str,
) -> dict[str, Any]:
    """
    Hybrid Tool: Extracts only the most critical ERROR logs for a service.
    
    Compresses thousands of raw logs into a top-3 summary of the worst
    errors happening on the specified service right now.
    """
    # We query the underlying deterministic tool for ERROR severity
    raw_result = csv_log_pattern_extract(engine, table="loki_logs", severity_filter="ERROR", min_count=1)
    
    patterns = raw_result.get("patterns", [])
    # Filter strictly to the cmdb_id requested
    service_patterns = [p for p in patterns if p.get("cmdb_id") == cmdb_filter]
    
    if not service_patterns:
        return {
            "status": "HEALTHY",
            "summary": f"No ERROR logs found for service '{cmdb_filter}'. The service is not crashing or throwing exceptions."
        }
        
    # Sort by occurrence count descending
    service_patterns.sort(key=lambda x: x.get("occurrence_count", 0), reverse=True)
    top_3 = service_patterns[:3]
    
    bullets = []
    for p in top_3:
        bullets.append(f"- '{p.get('pattern')}' (Occurred {p.get('occurrence_count')} times)")
        
    summary = f"Detected {len(service_patterns)} distinct ERROR patterns for '{cmdb_filter}'. Top critical errors:\n" + "\n".join(bullets)
    
    return {
        "status": "ANOMALOUS",
        "summary": summary
    }

# Hybrid registry export
HYBRID_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "csv_analyze_metric_health": {
        "fn": csv_analyze_metric_health,
        "description": "Checks a specific metric for anomalies and returns a compressed health summary. Do not use raw SQL for health checks.",
        "args": ["metric_kpi", "cmdb_filter?"],
    },
    "csv_extract_log_errors": {
        "fn": csv_extract_log_errors,
        "description": "Extracts the top 3 most frequent ERROR logs for a specific service. Use this instead of querying raw logs.",
        "args": ["cmdb_filter"],
    },
}
