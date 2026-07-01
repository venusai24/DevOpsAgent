"""Custom designed tools missing from the primary spec but required by architecture.

Designed Tools:
1. query_earliest_metric_anomaly: Needed by T0_map coverage guardrails (Section 9.3).
2. query_anomalous_traces: Needed to source trace_ids for build_span_tree_summary (Section 10.2).
3. extract_trace_dependency_edges: Mentioned in Appendix B.5 to populate refined_dependency_graph.
"""


import pandas as pd
from pydantic import BaseModel, Field

from ..base import BaseTool
from ..models import ToolContext, ToolMetadata, ToolSchema
from .baseline_tools import _BASELINE_STORE

# --- Schemas ---

class QueryEarliestMetricAnomalyInput(BaseModel):
    csv_path: str = Field(description="Path to the container_metrics.csv file.")
    cmdb_id: str = Field(description="The component to query.")
    baseline_ref: str = Field(description="Baseline reference string.")
    time_window_start: str = Field(description="Start timestamp.")
    time_window_end: str = Field(description="End timestamp.")

class QueryEarliestMetricAnomalyOutput(BaseModel):
    earliest_anomaly_timestamp: str | None = Field(description="ISO 8601 timestamp of the first breached metric.")
    kpi_name: str | None = Field(description="The metric that breached first.")

class QueryAnomalousTracesInput(BaseModel):
    traces_path: str = Field(description="Path to incident_traces.csv.")
    cmdb_id: str = Field(description="Target component.")
    time_window_start: str = Field(description="Start timestamp.")
    time_window_end: str = Field(description="End timestamp.")
    max_traces: int = Field(default=5)

class QueryAnomalousTracesOutput(BaseModel):
    trace_ids: list[str] = Field(description="List of anomalous trace IDs.")

class ExtractTraceDependencyEdgesInput(BaseModel):
    traces_path: str = Field(description="Path to incident_traces.csv.")
    trace_ids: list[str] = Field(description="List of trace IDs to extract topology edges from.")

class ExtractTraceDependencyEdgesOutput(BaseModel):
    edges: list[dict[str, str]] = Field(description="List of {caller: cmdb_id, callee: cmdb_id} pairs observed in traces.")

# --- Implementations ---

class QueryEarliestMetricAnomalyTool(BaseTool[QueryEarliestMetricAnomalyInput, QueryEarliestMetricAnomalyOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="query_earliest_metric_anomaly",
            description="Finds the exact onset timestamp of the first anomaly for a component to establish T0.",
            version="1.0.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryEarliestMetricAnomalyInput]:
        return ToolSchema(input_type=QueryEarliestMetricAnomalyInput, output_type=QueryEarliestMetricAnomalyOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryEarliestMetricAnomalyInput) -> QueryEarliestMetricAnomalyOutput:
        baseline_data = _BASELINE_STORE.get(inputs.baseline_ref, {}).get("container_metrics", {}).get(inputs.cmdb_id, {})
        if not baseline_data:
            return QueryEarliestMetricAnomalyOutput(earliest_anomaly_timestamp=None, kpi_name=None)
            
        try:
            df = pd.read_csv(inputs.csv_path)
            t_start = pd.to_datetime(inputs.time_window_start).timestamp()
            t_end = pd.to_datetime(inputs.time_window_end).timestamp()
        except Exception:
            return QueryEarliestMetricAnomalyOutput(earliest_anomaly_timestamp=None, kpi_name=None)

        df = df[(df['cmdb_id'] == inputs.cmdb_id) & (df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]
        
        earliest_ts = float('inf')
        earliest_kpi = None
        
        for kpi, group in df.groupby('kpi_name'):
            if kpi not in baseline_data:
                continue
            mean = baseline_data[kpi]['mean']
            std = baseline_data[kpi]['std']
            
            if std == 0:
                group['z_score'] = group['value'].apply(lambda x: 0 if x == mean else 3.0)
            else:
                group['z_score'] = (group['value'] - mean).abs() / std
                
            anomalous = group[group['z_score'] > 3.0]
            if not anomalous.empty:
                min_ts = anomalous['timestamp'].min()
                if min_ts < earliest_ts:
                    earliest_ts = min_ts
                    earliest_kpi = kpi
                    
        if earliest_ts == float('inf'):
            return QueryEarliestMetricAnomalyOutput(earliest_anomaly_timestamp=None, kpi_name=None)
            
        import datetime
        dt_str = datetime.datetime.fromtimestamp(earliest_ts).isoformat()
        return QueryEarliestMetricAnomalyOutput(earliest_anomaly_timestamp=dt_str, kpi_name=earliest_kpi)


class QueryAnomalousTracesTool(BaseTool[QueryAnomalousTracesInput, QueryAnomalousTracesOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="query_anomalous_traces",
            description="Queries the trace backend for anomalous trace IDs involving the target component.",
            version="1.0.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryAnomalousTracesInput]:
        return ToolSchema(input_type=QueryAnomalousTracesInput, output_type=QueryAnomalousTracesOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryAnomalousTracesInput) -> QueryAnomalousTracesOutput:
        try:
            df = pd.read_csv(inputs.traces_path)
        except Exception:
            return QueryAnomalousTracesOutput(trace_ids=[])
            
        # Segment by endpoint -> we use cmdb_id as endpoint proxy
        df_target = df[df['cmdb_id'] == inputs.cmdb_id].copy()
        if df_target.empty:
            return QueryAnomalousTracesOutput(trace_ids=[])
            
        # Absolute Latency Trigger: Duration > 500 ms (discard fast ones to avoid healthy system trap)
        absolute_floor = 500.0
        
        # Calculate Median and MAD on the target's spans
        median_dur = df_target['duration'].median()
        mad_dur = (df_target['duration'] - median_dur).abs().median()
        
        # Avoid MAD = 0
        if mad_dur == 0:
            mad_dur = 1.0 
            
        threshold = median_dur + (3 * mad_dur)
        
        # Anomalous traces are those passing both absolute floor and relative threshold
        anomalous_spans = df_target[(df_target['duration'] > absolute_floor) & (df_target['duration'] > threshold)]
        
        if anomalous_spans.empty:
            # Fallback to top durations if none are > threshold but maybe some are massive
            anomalous_spans = df_target[df_target['duration'] > absolute_floor]
            
        # Sort by severity (variance from MAD -> Total duration)
        anomalous_spans['variance'] = anomalous_spans['duration'] - median_dur
        sorted_spans = anomalous_spans.sort_values(by=['variance', 'duration'], ascending=[False, False])
        
        # Get top K unique trace IDs
        trace_ids = sorted_spans['trace_id'].unique().tolist()
        top_k = trace_ids[:inputs.max_traces]
        
        return QueryAnomalousTracesOutput(trace_ids=top_k)


class ExtractTraceDependencyEdgesTool(BaseTool[ExtractTraceDependencyEdgesInput, ExtractTraceDependencyEdgesOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="extract_trace_dependency_edges",
            description="Extracts deterministic topological edges from trace span parent-child relationships.",
            version="1.0.0"
        )
    @property
    def schema(self) -> ToolSchema[ExtractTraceDependencyEdgesInput]:
        return ToolSchema(input_type=ExtractTraceDependencyEdgesInput, output_type=ExtractTraceDependencyEdgesOutput)
    async def execute(self, ctx: ToolContext, inputs: ExtractTraceDependencyEdgesInput) -> ExtractTraceDependencyEdgesOutput:
        try:
            df = pd.read_csv(inputs.traces_path)
        except Exception:
            return ExtractTraceDependencyEdgesOutput(edges=[])
            
        if inputs.trace_ids:
            df = df[df['trace_id'].isin(inputs.trace_ids)]
            
        # parent_id -> caller, span_id -> callee
        # We join df with itself: df1 (caller) and df2 (callee) where df1.span_id == df2.parent_id
        merged = df.merge(df, left_on='span_id', right_on='parent_id', suffixes=('_caller', '_callee'))
        
        # Remove self edges if any
        merged = merged[merged['cmdb_id_caller'] != merged['cmdb_id_callee']]
        
        edges_set = set()
        for _, row in merged.iterrows():
            caller = row['cmdb_id_caller']
            callee = row['cmdb_id_callee']
            edges_set.add((caller, callee))
            
        result = [{"caller": edge[0], "callee": edge[1]} for edge in edges_set]
        return ExtractTraceDependencyEdgesOutput(edges=result)
