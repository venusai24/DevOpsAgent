"""Discovery Tools (Tools 2 and 3)."""

from pydantic import BaseModel, Field

from devops_agent.core.db.duckdb_client import DuckDBClient

from ..base import BaseTool
from ..models import ToolContext, ToolMetadata, ToolSchema
from .baseline_tools import _BASELINE_STORE

# --- Schemas ---

class QueryAnomalyOutput(BaseModel):
    affected_components: list[str] = Field(description="List of components (cmdb_id or tc) that breached the threshold.")
    anomalous_metrics_by_component: dict[str, list[str]] = Field(description="Mapping of component to a list of breaching metrics/logs.")
    t0_sources: list[str] = Field(description="Components that breached first temporally.")
    no_anomalies_detected: bool = Field(description="True if no components breached the criteria.")

class QueryMetricsInput(BaseModel):
    baseline_ref: str = Field(description="Reference key from compute_baseline_statistics.")
    time_range_start: str = Field(description="Incident start window.")
    time_range_end: str = Field(description="Incident end window.")
    anomaly_threshold_z: float = Field(default=3.0, description="Minimum z-score deviation to flag.")

class QueryAppStatsInput(BaseModel):
    baseline_ref: str = Field(description="Reference key from compute_baseline_statistics.")
    time_range_start: str = Field(description="Incident start window.")
    time_range_end: str = Field(description="Incident end window.")

class QueryLogsInput(BaseModel):
    baseline_ref: str = Field(description="Reference key from compute_baseline_statistics.")
    time_range_start: str = Field(description="Incident start window.")
    time_range_end: str = Field(description="Incident end window.")

class QueryTracesInput(BaseModel):
    baseline_ref: str = Field(description="Reference key from compute_baseline_statistics.")
    time_range_start: str = Field(description="Incident start window.")
    time_range_end: str = Field(description="Incident end window.")

class RunConnectedComponentAnalysisInput(BaseModel):
    affected_components: list[str] = Field(description="Output from query_anomalous_components.")
    max_degrees_of_separation: int = Field(default=2, description="How far to traverse for connected clusters.")

class RunConnectedComponentAnalysisOutput(BaseModel):
    primary_cluster: list[str] = Field(description="The largest connected graph of affected components.")
    concurrent_incident_clusters: list[list[str]] = Field(description="Disconnected sub-graphs experiencing simultaneous anomalies.")
    boundary_ambiguous: list[str] = Field(description="Components on the edge of the blast radius.")

class QueryMetricsTool(BaseTool[QueryMetricsInput, QueryAnomalyOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="triage_query_metrics",
            description="Identifies components with statistical KPI deviations against the baseline using container_metrics.csv.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryMetricsInput]:
        return ToolSchema(input_type=QueryMetricsInput, output_type=QueryAnomalyOutput)
        
    async def execute(self, ctx: ToolContext, inputs: QueryMetricsInput) -> QueryAnomalyOutput:
        baseline_data = _BASELINE_STORE.get(inputs.baseline_ref, {}).get("container_metrics", {})
        db = DuckDBClient.get_instance()
        
        try:
            import pandas as pd
            t_start = pd.to_datetime(inputs.time_range_start).timestamp()
            t_end = pd.to_datetime(inputs.time_range_end).timestamp()
        except ValueError:
            t_start = float(inputs.time_range_start)
            t_end = float(inputs.time_range_end)

        sql = f"SELECT * FROM read_csv_auto('{ctx.metrics_path}') WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
        df = db.query(sql, (t_start, t_end))
        
        anomalous_kpis = {}
        earliest_ts = float('inf')
        t0_components = set()

        for (cmdb_id, kpi_name), group in df.groupby(['cmdb_id', 'kpi_name']):
            if cmdb_id not in baseline_data or kpi_name not in baseline_data[cmdb_id]:
                continue
            
            mean = baseline_data[cmdb_id][kpi_name]['mean']
            std = baseline_data[cmdb_id][kpi_name]['std']
            
            if std == 0:
                group['z_score'] = group['value'].apply(lambda x: 0 if x == mean else 3.0)
            else:
                group['z_score'] = (group['value'] - mean).abs() / std
                
            anomalous = group[group['z_score'] > inputs.anomaly_threshold_z]
            if not anomalous.empty:
                if cmdb_id not in anomalous_kpis:
                    anomalous_kpis[cmdb_id] = []
                anomalous_kpis[cmdb_id].append(kpi_name)
                
                min_ts = anomalous['timestamp'].min()
                if min_ts < earliest_ts:
                    earliest_ts = min_ts
                    t0_components = {cmdb_id}
                elif min_ts == earliest_ts:
                    t0_components.add(cmdb_id)

        affected = list(anomalous_kpis.keys())
        return QueryAnomalyOutput(
            affected_components=affected,
            anomalous_metrics_by_component=anomalous_kpis,
            t0_sources=list(t0_components),
            no_anomalies_detected=len(affected) == 0
        )

class QueryAppStatsTool(BaseTool[QueryAppStatsInput, QueryAnomalyOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="triage_query_app_stats",
            description="Identifies anomalies in app stats. Does not have 'cmdb_id' or 'kpi_name'.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryAppStatsInput]:
        return ToolSchema(input_type=QueryAppStatsInput, output_type=QueryAnomalyOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryAppStatsInput) -> QueryAnomalyOutput:
        baseline_data = _BASELINE_STORE.get(inputs.baseline_ref, {}).get("app_metrics", {})
        db = DuckDBClient.get_instance()
        
        try:
            import pandas as pd
            t_start = pd.to_datetime(inputs.time_range_start).timestamp()
            t_end = pd.to_datetime(inputs.time_range_end).timestamp()
        except ValueError:
            t_start = float(inputs.time_range_start)
            t_end = float(inputs.time_range_end)

        sql = f"SELECT * FROM read_csv_auto('{ctx.app_stats_path}') WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
        df = db.query(sql, (t_start, t_end))
        
        anomalous_tcs = {}
        earliest_ts = float('inf')
        t0_components = set()
        metrics = ['rr', 'sr', 'cnt', 'mrt']

        for tc, group in df.groupby('tc'):
            if tc not in baseline_data:
                continue
            
            for m in metrics:
                if m not in group.columns: continue
                mean = baseline_data[tc][m]['mean']
                std = baseline_data[tc][m]['std']
                
                if std == 0:
                    group[f'z_score_{m}'] = group[m].apply(lambda x: 0 if x == mean else 3.0)
                else:
                    group[f'z_score_{m}'] = (group[m] - mean).abs() / std
                    
                anomalous = group[group[f'z_score_{m}'] > 2.5]
                if not anomalous.empty:
                    if tc not in anomalous_tcs:
                        anomalous_tcs[tc] = []
                    anomalous_tcs[tc].append(m)
                    
                    min_ts = anomalous['timestamp'].min()
                    if min_ts < earliest_ts:
                        earliest_ts = min_ts
                        t0_components = {tc}
                    elif min_ts == earliest_ts:
                        t0_components.add(tc)

        affected = list(anomalous_tcs.keys())
        return QueryAnomalyOutput(
            affected_components=affected,
            anomalous_metrics_by_component=anomalous_tcs,
            t0_sources=list(t0_components),
            no_anomalies_detected=len(affected) == 0
        )

class QueryLogsTool(BaseTool[QueryLogsInput, QueryAnomalyOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="triage_query_logs",
            description="Analyzes incident logs for anomalies.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryLogsInput]:
        return ToolSchema(input_type=QueryLogsInput, output_type=QueryAnomalyOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryLogsInput) -> QueryAnomalyOutput:
        try:
            import pandas as pd
            t_start = pd.to_datetime(inputs.time_range_start).timestamp()
            t_end = pd.to_datetime(inputs.time_range_end).timestamp()
        except ValueError:
            t_start = float(inputs.time_range_start)
            t_end = float(inputs.time_range_end)

        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT cmdb_id, timestamp, value 
            FROM read_csv_auto('{ctx.logs_path}') 
            WHERE timestamp >= ? AND timestamp <= ?
            AND (value LIKE '%ERROR%' OR value LIKE '%Exception%' OR value LIKE '%Failure%')
        """
        df = db.query(sql, (t_start, t_end))
        
        anomalous_cmdb = {}
        earliest_ts = float('inf')
        t0_components = set()
        
        for cmdb_id, group in df.groupby('cmdb_id'):
            anomalous_cmdb[cmdb_id] = ["ErrorLogsDetected"]
            min_ts = group['timestamp'].min()
            if min_ts < earliest_ts:
                earliest_ts = min_ts
                t0_components = {cmdb_id}
            elif min_ts == earliest_ts:
                t0_components.add(cmdb_id)
                
        affected = list(anomalous_cmdb.keys())
        return QueryAnomalyOutput(
            affected_components=affected,
            anomalous_metrics_by_component=anomalous_cmdb,
            t0_sources=list(t0_components),
            no_anomalies_detected=len(affected) == 0
        )

class QueryTracesTool(BaseTool[QueryTracesInput, QueryAnomalyOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="triage_query_traces",
            description="Analyzes incident traces for anomalous spans.",
            version="1.1.0"
        )
    @property
    def schema(self) -> ToolSchema[QueryTracesInput]:
        return ToolSchema(input_type=QueryTracesInput, output_type=QueryAnomalyOutput)
    async def execute(self, ctx: ToolContext, inputs: QueryTracesInput) -> QueryAnomalyOutput:
        try:
            import pandas as pd
            t_start = pd.to_datetime(inputs.time_range_start).timestamp() * 1000
            t_end = pd.to_datetime(inputs.time_range_end).timestamp() * 1000
        except ValueError:
            t_start = float(inputs.time_range_start) * 1000
            t_end = float(inputs.time_range_end) * 1000

        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT cmdb_id, timestamp, duration 
            FROM read_csv_auto('{ctx.traces_path}') 
            WHERE timestamp >= ? AND timestamp <= ?
            AND duration > 5000
        """
        df = db.query(sql, (t_start, t_end))
        
        anomalous_cmdb = {}
        earliest_ts = float('inf')
        t0_components = set()
        
        for cmdb_id, group in df.groupby('cmdb_id'):
            anomalous_cmdb[cmdb_id] = ["HighDurationTrace"]
            min_ts = group['timestamp'].min()
            if min_ts < earliest_ts:
                earliest_ts = min_ts
                t0_components = {cmdb_id}
            elif min_ts == earliest_ts:
                t0_components.add(cmdb_id)
                
        affected = list(anomalous_cmdb.keys())
        return QueryAnomalyOutput(
            affected_components=affected,
            anomalous_metrics_by_component=anomalous_cmdb,
            t0_sources=list(t0_components),
            no_anomalies_detected=len(affected) == 0
        )

class RunConnectedComponentAnalysisTool(BaseTool[RunConnectedComponentAnalysisInput, RunConnectedComponentAnalysisOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="run_connected_component_analysis",
            description="Groups anomalous components into connected sub-graphs based on the topology manifest.",
            version="1.0.0"
        )
    @property
    def schema(self) -> ToolSchema[RunConnectedComponentAnalysisInput]:
        return ToolSchema(input_type=RunConnectedComponentAnalysisInput, output_type=RunConnectedComponentAnalysisOutput)
    async def execute(self, ctx: ToolContext, inputs: RunConnectedComponentAnalysisInput) -> RunConnectedComponentAnalysisOutput:
        from collections import defaultdict, deque

        affected = set(inputs.affected_components)
        if not affected:
            return RunConnectedComponentAnalysisOutput(primary_cluster=[], concurrent_incident_clusters=[], boundary_ambiguous=[])

        topology = ctx.topology_graph or {}
        if not topology:
            # Graceful degradation if no topology graph is available
            return RunConnectedComponentAnalysisOutput(
                primary_cluster=inputs.affected_components,
                concurrent_incident_clusters=[],
                boundary_ambiguous=[]
            )

        # 1. Build undirected adjacency map
        undirected = defaultdict(set)
        for node, deps in topology.items():
            for dep in deps:
                undirected[node].add(dep)
                undirected[dep].add(node)

        # 2. Identify ghost nodes (anomalous components not in topology)
        ghost_nodes = {c for c in affected if c not in undirected}
        
        # 3. Identify boundary ambiguous nodes
        boundary_ambiguous = set()
        for node in affected - ghost_nodes:
            if any(n not in affected for n in undirected.get(node, set())):
                boundary_ambiguous.add(node)

        # 4. Build edges between affected nodes that are within max_degrees_of_separation
        affected_adj = defaultdict(set)

        for start_node in affected - ghost_nodes:
            queue = deque([(start_node, 0)])
            local_visited = {start_node}
            
            while queue:
                curr, depth = queue.popleft()
                
                if curr != start_node and curr in affected:
                    affected_adj[start_node].add(curr)
                    affected_adj[curr].add(start_node)
                
                if depth < inputs.max_degrees_of_separation:
                    for neighbor in undirected.get(curr, set()):
                        if neighbor not in local_visited:
                            local_visited.add(neighbor)
                            queue.append((neighbor, depth + 1))

        # 5. Find connected components in affected_adj
        clusters = []
        visited_affected = set()
        
        for node in affected - ghost_nodes:
            if node in visited_affected:
                continue
            
            comp = set()
            queue = deque([node])
            while queue:
                curr = queue.popleft()
                if curr not in visited_affected:
                    visited_affected.add(curr)
                    comp.add(curr)
                    for neighbor in affected_adj[curr]:
                        if neighbor not in visited_affected:
                            queue.append(neighbor)
            clusters.append(list(comp))
            
        # 6. Add ghost nodes as individual clusters
        for ghost in ghost_nodes:
            clusters.append([ghost])
            
        # Sort clusters by size (descending)
        clusters.sort(key=len, reverse=True)
        
        if not clusters:
            return RunConnectedComponentAnalysisOutput(primary_cluster=[], concurrent_incident_clusters=[], boundary_ambiguous=[])
            
        primary_cluster = clusters[0]
        concurrent = clusters[1:]
        
        return RunConnectedComponentAnalysisOutput(
            primary_cluster=primary_cluster,
            concurrent_incident_clusters=concurrent,
            boundary_ambiguous=list(boundary_ambiguous)
        )
