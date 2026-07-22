"""Tracing and Dependency Inference Tools (Tools 6, 7, 8)."""


from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from devops_agent.core.db.duckdb_client import DuckDBClient
from devops_agent.tools.interfaces.validators import parse_timestamp

from ..base import BaseTool
from ..models import ToolContext, ToolMetadata, ToolSchema

# --- Schemas ---

class QueryAnomalousTracesInput(BaseModel):
    time_range_start: str | int | float = Field(description="Incident start window (ISO 8601 or timestamp string).")
    time_range_end: str | int | float = Field(description="Incident end window (ISO 8601 or timestamp string).")
    baseline_ref: str = Field(description="Baseline reference key.")
    cmdb_ids: list[str] = Field(description="List of cmdb_ids to check. Must explicitly exist in the topology. Do not hallucinate component IDs.")

    @field_validator("time_range_start", "time_range_end", mode="before")
    @classmethod
    def _validate_timestamp(cls, v, info):
        return parse_timestamp(v, info.field_name)

class BottleneckSummary(BaseModel):
    trace_id: str
    cmdb_id: str
    span_id: str
    duration_ms: float
    status_code: str

class QueryAnomalousTracesOutput(BaseModel):
    anomalous_traces: list[BottleneckSummary]

class ExtractTraceDependencyEdgesInput(BaseModel):
    time_range_start: str | int | float
    time_range_end: str | int | float
    cmdb_ids: list[str]

    @field_validator("time_range_start", "time_range_end", mode="before")
    @classmethod
    def _validate_timestamp(cls, v, info):
        return parse_timestamp(v, info.field_name)

class ExtractTraceDependencyEdgesOutput(BaseModel):
    caller_callee_frequencies: dict[str, dict[str, int]] = Field(description="{caller -> {callee -> count}}")

class BuildSpanTreeSummaryInput(BaseModel):
    trace_ids: list[str] = Field(description="Trace IDs obtained from query_anomalous_traces.")

class BuildSpanTreeSummaryOutput(BaseModel):
    primary_bottleneck_cmdb_id: str | None = Field(description="The component where the majority of anomalous duration was spent.")
    trace_coverage_pct: float = Field(description="Percentage of the investigation cluster covered by these traces.")
    structural_anomalies: list[str] = Field(description="Missing spans, unexpected loops, etc.")

class RunPropagationDirectionCheckInput(BaseModel):
    caller_cmdb_id: str
    callee_cmdb_id: str
    baseline_ref: str

class RunPropagationDirectionCheckOutput(BaseModel):
    direction: str = Field(description="One of: 'forward', 'backward', 'simultaneous'.")
    lag_ms: float = Field(description="Time shift between the two components' anomaly onsets.")

class InferMetricDependencyEdgesInput(BaseModel):
    all_anomalous_components: list[str] = Field(description="cmdb_ids of all components.", min_length=2, max_length=50)
    evidence_available: dict[str, list[Any]] = Field(default_factory=dict, description="Optional evidence mapping.")
    T0_map: dict[str, str] = Field(description="Onset timestamps per component: {cmdb_id -> ISO 8601 UTC string}.")
    collection_interval: int = Field(description="Metric collection interval in seconds or minutes.")
    layer_map: dict[str, str] = Field(description="Component layer mapping.")
    baseline_ref: str

class InferMetricDependencyEdgesOutput(BaseModel):
    inferred_edges: list[dict[str, Any]] = Field(description="List of inferred edges.")
    components_without_inferred_edges: list[str] = Field(description="Components that could not be connected.")
    coverage_achieved: float = Field(description="Percentage of cluster connected by inferred edges.")
    inference_quality: str = Field(description="'high', 'acceptable', 'poor'")
    low_confidence_warning: str | None = Field(default=None, description="Optional warning if confidence is low.")

# --- Implementations ---

class QueryAnomalousTracesTool(BaseTool[QueryAnomalousTracesInput, QueryAnomalousTracesOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="query_anomalous_traces", description="Queries slow or error traces.", version="1.1.0")
    @property
    def schema(self) -> ToolSchema[QueryAnomalousTracesInput]:
        return ToolSchema(input_type=QueryAnomalousTracesInput, output_type=QueryAnomalousTracesOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryAnomalousTracesInput) -> QueryAnomalousTracesOutput:
        # Inputs are already valid floats due to validator
        t_start = inputs.time_range_start
        t_end = inputs.time_range_end

        placeholders = ','.join(['?'] * len(inputs.cmdb_ids))
        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT trace_id, cmdb_id, span_id, duration as duration_ms
            FROM read_csv_auto('{ctx.traces_path}')
            WHERE timestamp >= ? AND timestamp <= ?
            AND cmdb_id IN ({placeholders})
            AND (duration > 5000)
            LIMIT 50
        """
        df = db.query(sql, (t_start * 1000, t_end * 1000) + tuple(inputs.cmdb_ids))
        results = []
        for _, row in df.iterrows():
            results.append(BottleneckSummary(
                trace_id=str(row['trace_id']),
                cmdb_id=str(row['cmdb_id']),
                span_id=str(row['span_id']),
                duration_ms=float(row['duration_ms']),
                status_code="UNKNOWN"
            ))
        return QueryAnomalousTracesOutput(anomalous_traces=results)

class ExtractTraceDependencyEdgesTool(BaseTool[ExtractTraceDependencyEdgesInput, ExtractTraceDependencyEdgesOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="extract_trace_dependency_edges", description="Builds topology from traces.", version="1.1.0")
    @property
    def schema(self) -> ToolSchema[ExtractTraceDependencyEdgesInput]:
        return ToolSchema(input_type=ExtractTraceDependencyEdgesInput, output_type=ExtractTraceDependencyEdgesOutput)
    async def execute(self, ctx: ToolContext, inputs: ExtractTraceDependencyEdgesInput) -> ExtractTraceDependencyEdgesOutput:
        # Inputs are already valid floats due to validator
        t_start = inputs.time_range_start
        t_end = inputs.time_range_end

        db = DuckDBClient.get_instance()
        # Self-join on traces to find parent-child relationships
        sql = f"""
            SELECT parent.cmdb_id as caller, child.cmdb_id as callee, COUNT(*) as freq
            FROM read_csv_auto('{ctx.traces_path}') parent
            JOIN read_csv_auto('{ctx.traces_path}') child
              ON parent.trace_id = child.trace_id AND parent.span_id = child.parent_id
            WHERE parent.timestamp >= ? AND parent.timestamp <= ?
            GROUP BY parent.cmdb_id, child.cmdb_id
        """
        df = db.query(sql, (t_start * 1000, t_end * 1000))
        freqs = {}
        for _, row in df.iterrows():
            caller = str(row['caller'])
            callee = str(row['callee'])
            if caller not in freqs:
                freqs[caller] = {}
            freqs[caller][callee] = int(row['freq'])
        return ExtractTraceDependencyEdgesOutput(caller_callee_frequencies=freqs)

class BuildSpanTreeSummaryTool(BaseTool[BuildSpanTreeSummaryInput, BuildSpanTreeSummaryOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="build_span_tree_summary", description="Aggregates span data.", version="1.0.0")
    @property
    def schema(self) -> ToolSchema[BuildSpanTreeSummaryInput]:
        return ToolSchema(input_type=BuildSpanTreeSummaryInput, output_type=BuildSpanTreeSummaryOutput)
    async def execute(self, ctx: ToolContext, inputs: BuildSpanTreeSummaryInput) -> BuildSpanTreeSummaryOutput:
        if not inputs.trace_ids:
            return BuildSpanTreeSummaryOutput(primary_bottleneck_cmdb_id=None, trace_coverage_pct=0.0, structural_anomalies=[])
            
        db = DuckDBClient.get_instance()
        placeholders = ','.join(['?'] * len(inputs.trace_ids))
        sql = f"""
            SELECT cmdb_id, SUM(duration) as total_duration 
            FROM read_csv_auto('{ctx.traces_path}') 
            WHERE trace_id IN ({placeholders})
            GROUP BY cmdb_id
            ORDER BY total_duration DESC
        """
        df = db.query(sql, tuple(inputs.trace_ids))
        
        bottleneck = str(df.iloc[0]['cmdb_id']) if not df.empty else None
        return BuildSpanTreeSummaryOutput(primary_bottleneck_cmdb_id=bottleneck, trace_coverage_pct=1.0, structural_anomalies=[])

class RunPropagationDirectionCheckTool(BaseTool[RunPropagationDirectionCheckInput, RunPropagationDirectionCheckOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="run_propagation_direction_check", description="Fallback propagation.", version="1.0.0")
    @property
    def schema(self) -> ToolSchema[RunPropagationDirectionCheckInput]:
        return ToolSchema(input_type=RunPropagationDirectionCheckInput, output_type=RunPropagationDirectionCheckOutput)
    async def execute(self, ctx: ToolContext, inputs: RunPropagationDirectionCheckInput) -> RunPropagationDirectionCheckOutput:
        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT cmdb_id, MIN(timestamp) as min_ts 
            FROM read_csv_auto('{ctx.metrics_path}') 
            WHERE cmdb_id IN (?, ?) 
            GROUP BY cmdb_id
        """
        df = db.query(sql, (inputs.caller_cmdb_id, inputs.callee_cmdb_id))
        
        ts_map = df.set_index('cmdb_id')['min_ts'].to_dict()
        caller_ts = ts_map.get(inputs.caller_cmdb_id, float('inf'))
        callee_ts = ts_map.get(inputs.callee_cmdb_id, float('inf'))
        
        lag_ms = float((callee_ts - caller_ts) * 1000) if (caller_ts != float('inf') and callee_ts != float('inf')) else 0.0
        
        if lag_ms > 0:
            direction = "forward"
        elif lag_ms < 0:
            direction = "backward"
        else:
            direction = "simultaneous"
            
        return RunPropagationDirectionCheckOutput(direction=direction, lag_ms=lag_ms)

class InferMetricDependencyEdgesTool(BaseTool[InferMetricDependencyEdgesInput, InferMetricDependencyEdgesOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="infer_metric_dependency_edges", description="Fallback dependency edge.", version="1.0.0")
    @property
    def schema(self) -> ToolSchema[InferMetricDependencyEdgesInput]:
        return ToolSchema(input_type=InferMetricDependencyEdgesInput, output_type=InferMetricDependencyEdgesOutput)
    async def execute(self, ctx: ToolContext, inputs: InferMetricDependencyEdgesInput) -> InferMetricDependencyEdgesOutput:
        if len(inputs.all_anomalous_components) < 2:
            return InferMetricDependencyEdgesOutput(
                inferred_edges=[],
                components_without_inferred_edges=inputs.all_anomalous_components,
                coverage_achieved=0.0,
                inference_quality="poor",
                low_confidence_warning="Not enough components to infer edges."
            )
            
        db = DuckDBClient.get_instance()
        placeholders = ','.join(['?'] * len(inputs.all_anomalous_components))
        sql = f"""
            SELECT a.cmdb_id as caller_candidate, b.cmdb_id as callee_candidate, CORR(a.value, b.value) as r
            FROM read_csv_auto('{ctx.metrics_path}') a
            JOIN read_csv_auto('{ctx.metrics_path}') b
              ON a.timestamp = b.timestamp AND a.kpi_name = b.kpi_name
            WHERE a.cmdb_id != b.cmdb_id
              AND a.cmdb_id IN ({placeholders}) 
              AND b.cmdb_id IN ({placeholders})
            GROUP BY a.cmdb_id, b.cmdb_id
            HAVING ABS(CORR(a.value, b.value)) > 0.65
        """
        params = tuple(inputs.all_anomalous_components) * 2
        df = db.query(sql, params)
        
        inferred_edges = []
        connected_components = set()
        
        t0_parsed = {}
        for k, v in inputs.T0_map.items():
            if v:
                try:
                    t0_parsed[k] = pd.to_datetime(v).timestamp()
                except ValueError:
                    t0_parsed[k] = float(v)
        
        for _, row in df.iterrows():
            caller = str(row['caller_candidate'])
            callee = str(row['callee_candidate'])
            r_val = float(row['r'])
            
            t0_caller = t0_parsed.get(caller, float('inf'))
            t0_callee = t0_parsed.get(callee, float('inf'))
            
            if t0_callee <= t0_caller and t0_caller != float('inf'):
                confidence = round(min(0.95, abs(r_val)), 2)
                t0_gap = int(t0_caller - t0_callee) if t0_callee != float('inf') else 0
                
                inferred_edges.append({
                    "caller": caller,
                    "callee": callee,
                    "confidence": confidence,
                    "inference_basis": ["T0_ordering", "metric_co_movement"],
                    "pearson_r": round(r_val, 2),
                    "T0_gap_seconds": t0_gap,
                    "edge_tag": "metric_inferred"
                })
                connected_components.add(caller)
                connected_components.add(callee)
                
        unique_edges = {}
        for edge in inferred_edges:
            key = (edge["caller"], edge["callee"])
            if key not in unique_edges or unique_edges[key]["confidence"] < edge["confidence"]:
                unique_edges[key] = edge
                
        final_edges = list(unique_edges.values())
        isolated = list(set(inputs.all_anomalous_components) - connected_components)
        coverage = len(connected_components) / len(inputs.all_anomalous_components) if inputs.all_anomalous_components else 0.0
        
        if coverage >= 0.8:
            quality = "high"
        elif coverage >= 0.5:
            quality = "acceptable"
        else:
            quality = "poor"
            
        warning = None
        if quality == "poor":
            warning = "Low coverage achieved. Edges may not represent the full causal chain."
            
        return InferMetricDependencyEdgesOutput(
            inferred_edges=final_edges,
            components_without_inferred_edges=isolated,
            coverage_achieved=round(coverage, 2),
            inference_quality=quality,
            low_confidence_warning=warning
        )
