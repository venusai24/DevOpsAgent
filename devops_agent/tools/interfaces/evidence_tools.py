"""Evidence Gathering Tools (Tools 4 and 5)."""


from difflib import SequenceMatcher, get_close_matches

import pandas as pd
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, field_validator

from devops_agent.core.db.duckdb_client import DuckDBClient
from devops_agent.tools.exceptions import ToolMetricNotFoundError
from devops_agent.tools.interfaces.validators import parse_timestamp

from ..base import BaseTool
from ..models import ToolContext, ToolMetadata, ToolSchema
from .baseline_tools import _BASELINE_STORE

# --- Schemas ---

class QueryEarliestMetricAnomalyInput(BaseModel):
    time_range_start: str | int | float = Field(description="Incident start window.")
    time_range_end: str | int | float = Field(description="Incident end window.")
    baseline_ref: str = Field(description="Baseline reference key.")
    cmdb_ids: list[str] = Field(description="List of cmdb_ids to check.")

    @field_validator("time_range_start", "time_range_end", mode="before")
    @classmethod
    def _validate_timestamp(cls, v, info):
        return parse_timestamp(v, info.field_name)

class QueryEarliestMetricAnomalyOutput(BaseModel):
    t0_metrics: str | None = Field(description="Earliest timestamp an anomaly occurred across given cmdb_ids.")

class QueryMetricsInput(BaseModel):
    baseline_ref: str = Field(description="Baseline reference key.")
    hypothesis_id: str = Field(description="The name or ID of the hypothesis being tested.")
    cmdb_id: str = Field(description="The target component to query. Must be an exact match to a known cmdb_id.")
    kpi_name: str = Field(description="The specific resolved KPI name (not a generic category). Must explicitly exist in the schema. Do not guess or hallucinate metric names.")
    time_window_start: str | int | float = Field(description="Start time (ISO 8601 or timestamp string).")
    time_window_end: str | int | float = Field(description="End time (ISO 8601 or timestamp string).")

    @field_validator("time_window_start", "time_window_end", mode="before")
    @classmethod
    def _validate_timestamp(cls, v, info):
        return parse_timestamp(v, info.field_name)

class QueryMetricsOutput(BaseModel):
    evidence_id: str = Field(description="Unique ID for this evidence item.")
    hypothesis_id: str = Field(description="Ties back to the input hypothesis.")
    cmdb_id: str = Field(description="Target component.")
    max_z_score: float = Field(description="Maximum deviation during window.")
    anomaly_timestamp: str | None = Field(default=None, description="Timestamp of the maximum anomaly.")
    trend: str = Field(description="'spike', 'drop', 'flat', 'oscillating'")
    raw_data_points_summary: str = Field(description="A compact natural language summary of the curve.")
    resolution_note: str | None = Field(default=None, description="Note on how KPI was resolved.")

class QueryLogsInput(BaseModel):
    hypothesis_id: str = Field(description="The name or ID of the hypothesis being tested.")
    cmdb_id: str = Field(description="The target component to query. Must be an exact match to a known cmdb_id.")
    log_level_filter: str | None = Field(default="ERROR", description="Level to filter by (ERROR, WARN, etc). Do not guess non-standard levels.")
    grep_pattern: str | None = Field(default=None, description="Regex or keyword to search for.")
    time_window_start: str | int | float = Field(description="Start time (ISO 8601 or timestamp string).")
    time_window_end: str | int | float = Field(description="End time (ISO 8601 or timestamp string).")

    @field_validator("time_window_start", "time_window_end", mode="before")
    @classmethod
    def _validate_timestamp(cls, v, info):
        return parse_timestamp(v, info.field_name)

class QueryLogsOutput(BaseModel):
    evidence_id: str = Field(description="Unique ID for this evidence item.")
    hypothesis_id: str = Field(description="Ties back to the input hypothesis.")
    cmdb_id: str = Field(description="Target component.")
    match_count: int | str = Field(description="Total lines matching the criteria.")
    scanned_lines: int | None = Field(default=None)
    match_rate_pct: float | None = Field(default=None)
    sample_lines: list[str] = Field(description="Up to 5 representative log lines.")
    dominant_error_code: str | None = Field(default=None)
    note: str | None = Field(default=None)

# --- Helpers ---
AUTO_CORRECT_FLOOR = 0.90
AMBIGUITY_MARGIN = 0.10
SUGGESTION_CUTOFF = 0.60
MIN_PATTERN_LEN = 3
RESULT_CAP = 500

def resolve_kpi(cmdb_id: str, kpi_name: str, cmdb_baselines: dict) -> tuple[str, str | None]:
    host_metrics = cmdb_baselines.get(cmdb_id)
    if host_metrics is None:
        raise ToolException(
            f"Component '{cmdb_id}' has no telemetry data in the baseline "
            f"and should be skipped. "
            f"Known components with data: {list(cmdb_baselines.keys())[:10]}"
        )

    normalized = kpi_name.strip().lower()
    by_lower = {k.lower(): k for k in host_metrics}
    if normalized in by_lower:
        return by_lower[normalized], None

    ranked = sorted(by_lower.keys(), key=lambda k: SequenceMatcher(None, normalized, k).ratio(), reverse=True)
    if not ranked:
        raise ToolException(f"No metrics available for cmdb_id '{cmdb_id}'.")
        
    best_score = SequenceMatcher(None, normalized, ranked[0]).ratio()
    second_score = SequenceMatcher(None, normalized, ranked[1]).ratio() if len(ranked) > 1 else 0.0

    if best_score >= AUTO_CORRECT_FLOOR and (best_score - second_score) >= AMBIGUITY_MARGIN:
        resolved = by_lower[ranked[0]]
        return resolved, f"kpi_name '{kpi_name}' auto-resolved to '{resolved}' (similarity {best_score:.2f})"

    suggestions = get_close_matches(normalized, by_lower.keys(), n=5, cutoff=SUGGESTION_CUTOFF) or list(by_lower.keys())[:5]
    # Use ToolMetricNotFoundError (error_code="METRIC_NOT_FOUND") so the executor
    # treats this as a non-retryable semantic failure and stops immediately.
    raise ToolMetricNotFoundError(
        f"Metric '{kpi_name}' not found for {cmdb_id}. "
        f"Call list_available_metrics_for_component first to get exact names. "
        f"Closest available metrics: {[by_lower[s] for s in suggestions]}"
    )

# --- Implementations ---

class QueryEarliestMetricAnomalyTool(BaseTool[QueryEarliestMetricAnomalyInput, QueryEarliestMetricAnomalyOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="query_earliest_metric_anomaly",
            description="Finds earliest anomaly amongst specified components using DuckDB.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryEarliestMetricAnomalyInput]:
        return ToolSchema(input_type=QueryEarliestMetricAnomalyInput, output_type=QueryEarliestMetricAnomalyOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryEarliestMetricAnomalyInput) -> QueryEarliestMetricAnomalyOutput:
        # Simplification: DuckDB query across given ids
        # Inputs are already valid floats due to validator
        t_start = inputs.time_range_start
        t_end = inputs.time_range_end
            
        placeholders = ','.join(['?'] * len(inputs.cmdb_ids))
        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT MIN(timestamp) as earliest
            FROM read_csv_auto('{ctx.metrics_path}')
            WHERE timestamp >= ? AND timestamp <= ?
            AND cmdb_id IN ({placeholders})
        """
        df = db.query(sql, (t_start, t_end) + tuple(inputs.cmdb_ids))
        earliest = df['earliest'].iloc[0]
        return QueryEarliestMetricAnomalyOutput(t0_metrics=str(earliest) if pd.notna(earliest) else None)

class QueryMetricsTool(BaseTool[QueryMetricsInput, QueryMetricsOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="query_metrics_for_hypothesis",
            description="Fetches and summarizes metric time series for a specific component to validate a hypothesis using DuckDB.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryMetricsInput]:
        return ToolSchema(input_type=QueryMetricsInput, output_type=QueryMetricsOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryMetricsInput) -> QueryMetricsOutput:
        baseline_data = _BASELINE_STORE.get(inputs.baseline_ref, {}).get("container_metrics", {})
        
        resolved_kpi, note = resolve_kpi(inputs.cmdb_id, inputs.kpi_name, baseline_data)
        
        # Inputs are already valid floats due to validator
        t_start = inputs.time_window_start
        t_end = inputs.time_window_end
            
        db = DuckDBClient.get_instance()
        global_trend = "flat"

        cmdb_baselines = baseline_data.get(inputs.cmdb_id, {})
        cmdb_baseline = cmdb_baselines[resolved_kpi]
        mean = float(cmdb_baseline['mean'])
        std = float(cmdb_baseline['std'])
        
        if std == 0:
            z_expr = f"CASE WHEN value = {mean} THEN 0 ELSE 3.0 END"
        else:
            z_expr = f"ABS(value - {mean}) / {std}"
            
        sql = f"""
            SELECT timestamp, value, {z_expr} as z_score
            FROM read_csv_auto('{ctx.metrics_path}')
            WHERE timestamp >= ? AND timestamp <= ?
            AND cmdb_id = ?
            AND kpi_name = ?
            ORDER BY timestamp ASC
        """
        df = db.query(sql, (t_start, t_end, inputs.cmdb_id, resolved_kpi))
            
        if df.empty:
            return QueryMetricsOutput(evidence_id="ev_empty", hypothesis_id=inputs.hypothesis_id, cmdb_id=inputs.cmdb_id, max_z_score=0.0, anomaly_timestamp=None, trend="flat", raw_data_points_summary="No data in time window.", resolution_note=note)
            
        max_z_idx = df['z_score'].idxmax()
        max_z = float(df.loc[max_z_idx, 'z_score'])
        anomaly_ts = str(df.loc[max_z_idx, 'timestamp']) if max_z > 3.0 else None
        
        vals = df['value'].tolist()
        
        if max_z > 3.0:
            global_trend = "spike" if vals[-1] > vals[0] else "drop"
        else:
            global_trend = "flat"
                
        summary = f"{resolved_kpi} ({len(vals)} pts, max: {max(vals):.2f}, min: {min(vals):.2f}, max_z: {max_z:.2f})"
        evidence_id = f"ev_metric_{inputs.cmdb_id}_{resolved_kpi.replace('.', '_')}"
        return QueryMetricsOutput(evidence_id=evidence_id, hypothesis_id=inputs.hypothesis_id, cmdb_id=inputs.cmdb_id, max_z_score=max_z, anomaly_timestamp=anomaly_ts, trend=global_trend, raw_data_points_summary=summary, resolution_note=note)

class QueryLogsTool(BaseTool[QueryLogsInput, QueryLogsOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="query_logs_for_hypothesis",
            description="Scans and summarizes log lines for a component to validate a hypothesis using DuckDB.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryLogsInput]:
        return ToolSchema(input_type=QueryLogsInput, output_type=QueryLogsOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryLogsInput) -> QueryLogsOutput:
        db = DuckDBClient.get_instance()
        pattern = inputs.grep_pattern.strip() if inputs.grep_pattern else inputs.log_level_filter
        
        t_start = inputs.time_window_start
        t_end = inputs.time_window_end
        
        note = None
        exact = True
        
        if pattern:
            if len(pattern) < MIN_PATTERN_LEN:
                raise ToolException(
                    f"grep_pattern '{pattern}' is only {len(pattern)} character(s). "
                    f"Use at least {MIN_PATTERN_LEN}, ideally a specific token "
                    f"(error code, exception class, request id) rather than a common word."
                )
            
            sql = f"""
                SELECT value 
                FROM read_csv_auto('{ctx.logs_path}')
                WHERE cmdb_id = ?
                AND timestamp >= ? AND timestamp <= ?
                AND value ILIKE ?
                LIMIT ?
            """
            df = db.query(sql, (inputs.cmdb_id, t_start, t_end, f'%{pattern}%', RESULT_CAP + 1))
            
            total_sql = f"SELECT count(*) AS n FROM read_csv_auto('{ctx.logs_path}') WHERE cmdb_id = ? AND timestamp >= ? AND timestamp <= ?"
            total = db.query(total_sql, (inputs.cmdb_id, t_start, t_end))["n"][0]
            
            exact = len(df) <= RESULT_CAP
            match_count = len(df) if exact else f"{RESULT_CAP}+"
            rate_pct = round(min(len(df), RESULT_CAP) / total * 100, 2) if total else None
            
            if not exact:
                note = (
                    f"Capped at {RESULT_CAP}; this pattern is not selective. An exact count "
                    f"would add no diagnostic signal -- narrow the term or add a structured "
                    f"filter for a precise result."
                )
        else:
            sql = f"""
                SELECT value 
                FROM read_csv_auto('{ctx.logs_path}')
                WHERE cmdb_id = ?
                AND timestamp >= ? AND timestamp <= ?
            """
            df = db.query(sql, (inputs.cmdb_id, t_start, t_end))
            match_count = len(df)
            total = len(df)
            rate_pct = 100.0
            
        samples = df['value'].head(5).tolist()
        pattern_tag = (pattern or "all").replace(" ", "_")
        evidence_id = f"ev_log_{inputs.cmdb_id}_{pattern_tag}"
        return QueryLogsOutput(
            evidence_id=evidence_id, 
            hypothesis_id=inputs.hypothesis_id, 
            cmdb_id=inputs.cmdb_id, 
            match_count=match_count,
            scanned_lines=total,
            match_rate_pct=rate_pct,
            note=note,
            sample_lines=samples
        )

# ── Fix 2: Guard-rail tool so the LLM can discover exact metric names ───────

class ListAvailableMetricsInput(BaseModel):
    baseline_ref: str = Field(description="Baseline reference key from compute_baseline_statistics.")
    cmdb_id: str = Field(description="The component to list metrics for. Must be an exact cmdb_id.")

class ListAvailableMetricsOutput(BaseModel):
    cmdb_id: str = Field(description="The queried component.")
    available_metrics: list[str] = Field(
        description=(
            "Exact kpi_name strings that exist for this component. "
            "Use one of these verbatim as the kpi_name argument to query_metrics_for_hypothesis."
        )
    )
    metric_count: int = Field(description="Total number of available metrics.")

class ListAvailableMetricsTool(BaseTool[ListAvailableMetricsInput, ListAvailableMetricsOutput]):
    """Guard-rail tool: returns the exact kpi_name strings available for a component.

    The RCA LLM MUST call this before query_metrics_for_hypothesis whenever it
    wants to query a metric by a generic name (e.g. 'cpu', 'disk').  The returned
    names are the verbatim strings required by that tool.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="list_available_metrics_for_component",
            description=(
                "Returns the exact kpi_name strings available in the baseline for a given cmdb_id. "
                "ALWAYS call this before query_metrics_for_hypothesis if you are not certain of the "
                "exact metric name. Never use generic names like 'cpu' or 'disk' — they will fail."
            ),
            version="1.0.0"
        )

    @property
    def schema(self) -> ToolSchema[ListAvailableMetricsInput]:
        return ToolSchema(
            input_type=ListAvailableMetricsInput,
            output_type=ListAvailableMetricsOutput
        )

    async def execute(
        self, ctx: ToolContext, inputs: ListAvailableMetricsInput
    ) -> ListAvailableMetricsOutput:
        baseline_data = _BASELINE_STORE.get(inputs.baseline_ref, {}).get("container_metrics", {})
        host_metrics = baseline_data.get(inputs.cmdb_id)
        if host_metrics is None:
            # Component is in the declared topology but has no telemetry data
            # (e.g. ServiceTest* dummy nodes). Return an empty list so the
            # agent knows to skip this component rather than crashing.
            return ListAvailableMetricsOutput(
                cmdb_id=inputs.cmdb_id,
                available_metrics=[],
                metric_count=0,
            )
        metric_names = list(host_metrics.keys())
        return ListAvailableMetricsOutput(
            cmdb_id=inputs.cmdb_id,
            available_metrics=metric_names,
            metric_count=len(metric_names),
        )
