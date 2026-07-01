"""Evidence Gathering Tools (Tools 4 and 5)."""


import pandas as pd
from pydantic import BaseModel, Field

from devops_agent.core.db.duckdb_client import DuckDBClient

from ..base import BaseTool
from ..models import ToolContext, ToolMetadata, ToolSchema
from .baseline_tools import _BASELINE_STORE

# --- Schemas ---

class QueryEarliestMetricAnomalyInput(BaseModel):
    time_range_start: str = Field(description="Incident start window.")
    time_range_end: str = Field(description="Incident end window.")
    baseline_ref: str = Field(description="Baseline reference key.")
    cmdb_ids: list[str] = Field(description="List of cmdb_ids to check.")

class QueryEarliestMetricAnomalyOutput(BaseModel):
    t0_metrics: str | None = Field(description="Earliest timestamp an anomaly occurred across given cmdb_ids.")

class QueryMetricsInput(BaseModel):
    baseline_ref: str = Field(description="Baseline reference key.")
    hypothesis_id: str = Field(description="The name or ID of the hypothesis being tested.")
    cmdb_id: str = Field(description="The target component to query. Must be an exact match to a known cmdb_id.")
    kpi_name: str = Field(description="The specific resolved KPI name (not a generic category). Must explicitly exist in the schema. Do not guess or hallucinate metric names.")
    time_window_start: str = Field(description="Start time (ISO 8601 or timestamp string).")
    time_window_end: str = Field(description="End time (ISO 8601 or timestamp string).")

class QueryMetricsOutput(BaseModel):
    evidence_id: str = Field(description="Unique ID for this evidence item.")
    hypothesis_id: str = Field(description="Ties back to the input hypothesis.")
    cmdb_id: str = Field(description="Target component.")
    max_z_score: float = Field(description="Maximum deviation during window.")
    trend: str = Field(description="'spike', 'drop', 'flat', 'oscillating'")
    raw_data_points_summary: str = Field(description="A compact natural language summary of the curve.")

class QueryLogsInput(BaseModel):
    hypothesis_id: str = Field(description="The name or ID of the hypothesis being tested.")
    cmdb_id: str = Field(description="The target component to query. Must be an exact match to a known cmdb_id.")
    log_level_filter: str | None = Field(default="ERROR", description="Level to filter by (ERROR, WARN, etc). Do not guess non-standard levels.")
    grep_pattern: str | None = Field(default=None, description="Regex or keyword to search for.")

class QueryLogsOutput(BaseModel):
    evidence_id: str = Field(description="Unique ID for this evidence item.")
    hypothesis_id: str = Field(description="Ties back to the input hypothesis.")
    cmdb_id: str = Field(description="Target component.")
    match_count: int = Field(description="Total lines matching the criteria.")
    sample_lines: list[str] = Field(description="Up to 5 representative log lines.")
    dominant_error_code: str | None = Field(default=None)

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
        try:
            t_start = pd.to_datetime(inputs.time_range_start).timestamp()
            t_end = pd.to_datetime(inputs.time_range_end).timestamp()
        except ValueError:
            t_start = float(inputs.time_range_start)
            t_end = float(inputs.time_range_end)
            
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
        cmdb_baselines = baseline_data.get(inputs.cmdb_id, {})
        
        # Find all actual KPIs that match the requested category/name (case-insensitive)
        matching_kpis = [
            k for k in cmdb_baselines.keys()
            if inputs.kpi_name.lower() in k.lower() or k.lower() in inputs.kpi_name.lower()
        ]
        
        if not matching_kpis:
            return QueryMetricsOutput(evidence_id="ev_none", hypothesis_id=inputs.hypothesis_id, cmdb_id=inputs.cmdb_id, max_z_score=0.0, trend="flat", raw_data_points_summary="No baseline for this KPI category.")
            
        try:
            t_start = pd.to_datetime(inputs.time_window_start).timestamp()
            t_end = pd.to_datetime(inputs.time_window_end).timestamp()
        except ValueError:
            t_start = float(inputs.time_window_start)
            t_end = float(inputs.time_window_end)
            
        db = DuckDBClient.get_instance()
        global_max_z = 0.0
        global_trend = "flat"
        global_summary_parts = []
        
        for actual_kpi in matching_kpis:
            cmdb_baseline = cmdb_baselines[actual_kpi]
            mean = float(cmdb_baseline['mean'])
            std = float(cmdb_baseline['std'])
            
            if std == 0:
                z_expr = f"CASE WHEN value = {mean} THEN 0 ELSE 3.0 END"
            else:
                z_expr = f"ABS(value - {mean}) / {std}"
                
            sql = f"""
                SELECT value, {z_expr} as z_score
                FROM read_csv_auto('{ctx.metrics_path}')
                WHERE timestamp >= ? AND timestamp <= ?
                AND cmdb_id = ?
                AND kpi_name = ?
                ORDER BY timestamp ASC
            """
            df = db.query(sql, (t_start, t_end, inputs.cmdb_id, actual_kpi))
                
            if df.empty:
                continue
                
            max_z = float(df['z_score'].max())
            vals = df['value'].tolist()
            
            if max_z > global_max_z:
                global_max_z = max_z
                if max_z > 3.0:
                    global_trend = "spike" if vals[-1] > vals[0] else "drop"
                else:
                    global_trend = "flat"
                    
            summary_part = f"{actual_kpi} ({len(vals)} pts, max: {max(vals):.2f}, min: {min(vals):.2f}, max_z: {max_z:.2f})"
            global_summary_parts.append(summary_part)
            
        if not global_summary_parts:
            return QueryMetricsOutput(evidence_id="ev_empty", hypothesis_id=inputs.hypothesis_id, cmdb_id=inputs.cmdb_id, max_z_score=0.0, trend="flat", raw_data_points_summary="No data in time window.")
            
        summary = " | ".join(global_summary_parts)
        # Truncate summary if too long to save context
        if len(summary) > 500:
            summary = summary[:497] + "..."
            
        return QueryMetricsOutput(evidence_id="ev_metrics_01", hypothesis_id=inputs.hypothesis_id, cmdb_id=inputs.cmdb_id, max_z_score=global_max_z, trend=global_trend, raw_data_points_summary=summary)

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
        pattern = inputs.grep_pattern if inputs.grep_pattern else inputs.log_level_filter
        
        if pattern:
            # DuckDB ILIKE for pattern matching
            sql = f"""
                SELECT value 
                FROM read_csv_auto('{ctx.logs_path}')
                WHERE cmdb_id = ?
                AND value ILIKE ?
            """
            df = db.query(sql, (inputs.cmdb_id, f'%{pattern}%'))
        else:
            sql = f"""
                SELECT value 
                FROM read_csv_auto('{ctx.logs_path}')
                WHERE cmdb_id = ?
            """
            
            df = db.query(sql, (inputs.cmdb_id,))
        count = len(df)
        samples = df['value'].head(5).tolist()
        return QueryLogsOutput(evidence_id="ev_logs_01", hypothesis_id=inputs.hypothesis_id, cmdb_id=inputs.cmdb_id, match_count=count, sample_lines=samples)
