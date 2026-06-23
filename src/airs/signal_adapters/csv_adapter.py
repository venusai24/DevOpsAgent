"""
CSV Signal Adapter — Phase C.

Implements the SignalAdapter protocol for CSV telemetry files.

Unlike the other signal adapters (Metrics, Logs, Traces), this adapter:
  - Does NOT make network MCP calls
  - Instead, routes to the local DuckDB CsvEngine
  - Returns tool outputs as FilteredSignalPayload
  - Never injects raw rows into the context

The adapter acts as a dispatcher:
  tool_name → specific csv_engine tool function → FilteredSignalPayload

Context Hygiene Invariant:
  The pre_filter() step enforces that only statistical insights
  (anomaly lists, pattern counts, aggregated buckets) enter the
  EvidenceCandidate. Raw CSV rows are never propagated.

Architecture:
  ExecutionIntent (signal_type=CSV, tool_name='csv_anomaly_detection')
      → CsvSignalAdapter.execute()
          → CsvEngine.initialize()  [lazy — only on first call]
          → csv_anomaly_detection(engine, **intent.tool_spec.arguments)
      → CsvSignalAdapter.pre_filter()
          → enforce token budget, validate output structure
      → CsvSignalAdapter.normalize()
          → EvidenceCandidate(signal_source=SignalSource.CSV)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airs.csv_engine.engine import CsvEngine
from airs.csv_engine.tools import (
    csv_anomaly_detection,
    csv_correlation_analysis,
    csv_log_pattern_extract,
    csv_schema_discovery,
    csv_slow_span_detection,
    csv_sql_query,
    csv_time_window_summary,
)
from airs.models.intents import ToolSpec, SignalType
from airs.signal_adapters.base import (
    EvidenceCandidate,
    FilteredSignalPayload,
    RawSignalPayload,
    SignalAdapter,
)

log = logging.getLogger(__name__)

# Where the GUI writes the 5 CSV files.
# Default: incident_data/ relative to working directory.
# Override via CSV_INCIDENT_DATA_DIR environment variable.
_DEFAULT_DATA_DIR = Path(os.getenv("CSV_INCIDENT_DATA_DIR", "incident_data"))

# Max token estimate for a single CSV tool result (~4 chars per token)
_MAX_TOKENS = 2000
_MAX_CHARS = _MAX_TOKENS * 4

# Mapping: tool_name → callable(engine, **kwargs) → dict
_TOOL_DISPATCH: dict[str, Any] = {
    "csv_schema_discovery": lambda eng, **kw: csv_schema_discovery(eng),
    "csv_sql_query": lambda eng, **kw: csv_sql_query(eng, kw.get("query", "SELECT 1")),
    "csv_anomaly_detection": lambda eng, **kw: csv_anomaly_detection(
        eng,
        kpi_filter=kw.get("kpi_filter"),
        cmdb_filter=kw.get("cmdb_filter"),
    ),
    "csv_log_pattern_extract": lambda eng, **kw: csv_log_pattern_extract(
        eng,
        table=kw.get("table", "loki_logs"),
        severity_filter=kw.get("severity_filter"),
        min_count=kw.get("min_count", 2),
    ),
    "csv_slow_span_detection": lambda eng, **kw: csv_slow_span_detection(
        eng,
        table=kw.get("table", "jaeger_traces"),
        cmdb_filter=kw.get("cmdb_filter"),
        percentile_threshold=kw.get("percentile_threshold", 95.0),
    ),
    "csv_time_window_summary": lambda eng, **kw: csv_time_window_summary(
        eng,
        table=kw.get("table", "prometheus_metrics"),
        time_col=kw.get("time_col", "timestamp"),
        value_col=kw.get("value_col", "value"),
        window_minutes=kw.get("window_minutes", 5),
        kpi_filter=kw.get("kpi_filter"),
    ),
    "csv_correlation_analysis": lambda eng, **kw: csv_correlation_analysis(
        eng,
        metric_kpi=kw.get("metric_kpi", ""),
        log_pattern_prefix=kw.get("log_pattern_prefix", "ERROR"),
        window_minutes=kw.get("window_minutes", 5),
    ),
}


class CsvSignalAdapter(SignalAdapter):
    """
    Signal adapter for CSV telemetry files exported by the GUI module.

    The adapter maintains a lazily-initialized CsvEngine singleton per
    data directory. The engine is shared across all tools within the same
    investigation to avoid redundant DuckDB view registration.
    """

    signal_type: SignalType = SignalType.CSV

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DEFAULT_DATA_DIR
        self._engine: CsvEngine | None = None  # Lazy init

    def _get_engine(self) -> CsvEngine:
        """Return (or initialize) the shared CsvEngine for this investigation."""
        if self._engine is None:
            log.info("Initializing CsvEngine from: %s", self._data_dir)
            self._engine = CsvEngine(self._data_dir)
            schema = self._engine.initialize()
            log.info(
                "CsvEngine ready: %d tables registered, %d files missing",
                len(schema["registered_tables"]),
                len(schema.get("missing_files", [])),
            )
        return self._engine

    async def execute(
        self,
        tool_spec: ToolSpec,
        mcp_client: Any = None,  # Unused — CSV tools are local
    ) -> RawSignalPayload:
        """
        Dispatch to the appropriate CsvEngine tool.

        Args:
            tool_spec: ToolSpec with tool_name and arguments holding the parameters.
            mcp_client: Not used. CSV tools run in-process.

        Returns:
            RawSignalPayload wrapping the tool's JSON output.
        """
        tool_name = tool_spec.tool_name
        arguments = tool_spec.arguments

        if tool_name not in _TOOL_DISPATCH:
            return _build_error_raw(
                tool_name=tool_name,
                error=f"Unknown CSV tool: '{tool_name}'. "
                      f"Valid tools: {list(_TOOL_DISPATCH.keys())}",
            )

        engine = self._get_engine()

        try:
            result = _TOOL_DISPATCH[tool_name](engine, **arguments)
        except Exception as e:
            log.exception("CSV tool %s raised exception: %s", tool_name, e)
            return _build_error_raw(tool_name=tool_name, error=str(e))

        raw_text = json.dumps(result, default=str)
        digest = hashlib.sha256(raw_text.encode()).hexdigest()
        args_hash = hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest()

        return RawSignalPayload(
            server_id="csv_engine",
            tool_name=tool_name,
            content=result,
            raw_text=raw_text,
            raw_digest=digest,
            idempotency_key=f"csv:{tool_name}:{args_hash}",
            arguments_hash=args_hash,
            success=result.get("error") is None,
            error_code=None,
            error_message=result.get("error"),
        )

    def pre_filter(
        self,
        payload: RawSignalPayload,
        incident_service: str | None = None,
        incident_namespace: str | None = None,
    ) -> FilteredSignalPayload:
        """
        Validate and trim the tool output to enforce the token budget.

        The pre_filter does NOT inject raw CSV rows. It enforces:
          - Tool error check → flag error in filtered content
          - Token cap: if output > _MAX_CHARS, truncate and log a warning
          - Extract a concise summary field for quick LLM reference

        Returns:
            FilteredSignalPayload with filtered_content = compact insight dict.
        """
        content = payload.content or {}
        error = content.get("error") or payload.error_message

        if error:
            filtered_content = {
                "success": False,
                "tool": payload.tool_name,
                "error": error,
            }
            token_estimate = 30
        else:
            # Compact the content to stay within token budget
            compact = _compact_tool_output(payload.tool_name, content)
            raw_json = json.dumps(compact, default=str)

            if len(raw_json) > _MAX_CHARS:
                log.warning(
                    "CSV tool output for %s exceeded token budget (%d chars > %d). Trimming.",
                    payload.tool_name, len(raw_json), _MAX_CHARS,
                )
                compact = _trim_to_budget(compact, _MAX_CHARS)
                raw_json = json.dumps(compact, default=str)

            filtered_content = compact
            token_estimate = max(1, len(raw_json) // 4)

        return FilteredSignalPayload(
            server_id=payload.server_id,
            tool_name=payload.tool_name,
            filtered_content=filtered_content,
            token_estimate=token_estimate,
            filter_stats={
                "success": error is None,
                "tool_name": payload.tool_name,
                "rows_returned": content.get("row_count", content.get("anomaly_count", 0)),
                "truncated": content.get("truncated", False),
            },
            raw_digest=payload.raw_digest,
            idempotency_key=payload.idempotency_key,
            arguments_hash=payload.arguments_hash,
            received_at=datetime.now(timezone.utc),
        )

    def normalize(
        self,
        filtered: FilteredSignalPayload,
        hop_index: int = 0,
    ) -> EvidenceCandidate:
        """
        Wrap the filtered CSV insight into a standard EvidenceCandidate.

        The entity is the CSV table/tool that produced the evidence.
        The raw_identifiers list contains the table names that were queried.
        """
        from airs.models.evidence import EntityType, SignalSource

        return EvidenceCandidate(
            entity_type=EntityType.SERVICE,
            raw_identifiers=_extract_identifiers(filtered.filtered_content),
            signal_source=SignalSource.CSV,
            filtered_content=filtered.filtered_content,
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compact_tool_output(tool_name: str, content: dict) -> dict:
    """
    Apply tool-specific compaction rules to reduce token usage.
    Preserves all insight fields; removes bulk data arrays when possible.
    """
    if tool_name == "csv_schema_discovery":
        # Keep full schema — it's small and needed for planning
        return content

    if tool_name == "csv_anomaly_detection":
        # Keep anomalies list (already filtered to outliers only)
        return {
            "anomaly_count": content.get("anomaly_count", 0),
            "kpis_analyzed": content.get("kpis_analyzed", 0),
            "zscore_threshold": content.get("zscore_threshold"),
            "anomalies": content.get("anomalies", []),
            "truncated": content.get("truncated", False),
        }

    if tool_name == "csv_log_pattern_extract":
        # Keep pattern list (already deduplicated)
        return {
            "unique_patterns": content.get("unique_patterns", 0),
            "total_log_lines": content.get("total_log_lines", 0),
            "patterns": content.get("patterns", []),
            "truncated": content.get("truncated", False),
        }

    if tool_name == "csv_slow_span_detection":
        return {
            "slow_span_count": content.get("slow_span_count", 0),
            "services_analyzed": content.get("services_analyzed", 0),
            "percentile_threshold": content.get("percentile_threshold"),
            "slow_spans": content.get("slow_spans", []),
        }

    if tool_name == "csv_correlation_analysis":
        return {
            "correlation": content.get("correlation"),
            "interpretation": content.get("interpretation", ""),
            "sample_points": content.get("sample_points", 0),
        }

    # For csv_sql_query and csv_time_window_summary — return as-is
    return content


def _trim_to_budget(content: dict, max_chars: int) -> dict:
    """
    Trim list fields in a content dict to fit within max_chars.
    Trims the largest list field first until the JSON fits.
    """
    for key in ["anomalies", "patterns", "slow_spans", "rows", "buckets"]:
        if key in content and isinstance(content[key], list):
            items = content[key]
            while items and len(json.dumps(content, default=str)) > max_chars:
                items.pop()
            content[key] = items
            content["_trimmed"] = True
            if len(json.dumps(content, default=str)) <= max_chars:
                break
    return content


def _extract_identifiers(content: dict) -> list[str]:
    """Extract CMDB IDs or table names from tool output for entity resolution."""
    identifiers = set()
    for key in ["cmdb_id", "service", "source_file"]:
        if key in content and content[key]:
            identifiers.add(str(content[key]))
    # From lists of results
    for key in ["anomalies", "patterns", "slow_spans", "rows"]:
        for row in content.get(key, [])[:5]:
            if isinstance(row, dict) and "cmdb_id" in row:
                identifiers.add(str(row["cmdb_id"]))
    return list(identifiers) if identifiers else ["csv_engine"]


def _build_error_raw(tool_name: str, error: str) -> RawSignalPayload:
    return RawSignalPayload(
        server_id="csv_engine",
        tool_name=tool_name,
        content={"error": error},
        raw_text=json.dumps({"error": error}),
        raw_digest="",
        idempotency_key=f"csv:{tool_name}:error",
        arguments_hash="",
        success=False,
        error_code=None,
        error_message=error,
    )
