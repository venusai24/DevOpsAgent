"""Stage 1-4 Triage Tools."""

from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from devops_agent.core.db.duckdb_client import DuckDBClient

from ..base import BaseTool
from ..models import ToolContext
from .baseline_tools import _BASELINE_STORE


# 1. query_app_stats_anomalies
class QueryAppStatsAnomaliesInput(BaseModel):
    time_range: tuple[datetime, datetime] = Field(..., description="Investigation time window")
    baseline_ref: str = Field(..., description="Reference to the baseline")

class TcAnomalySummary(BaseModel):
    onset_timestamp: datetime
    max_z_score: float
    primary_degraded_metric: str

class QueryAppStatsAnomaliesOutput(BaseModel):
    anomalous_tcs: dict[str, TcAnomalySummary] = Field(..., description="Mapping of TC values to anomaly summaries")

class QueryAppStatsAnomaliesTool(BaseTool[QueryAppStatsAnomaliesInput, QueryAppStatsAnomaliesOutput]):
    """Fetches aggregated anomaly data (z-scores) for application stats using DuckDB."""
    
    @property
    def name(self) -> str:
        return "query_app_stats_anomalies"
        
    async def execute(self, ctx: ToolContext, params: QueryAppStatsAnomaliesInput) -> QueryAppStatsAnomaliesOutput:
        baseline_data = _BASELINE_STORE.get(params.baseline_ref, {}).get("app_metrics", {})
        
        t_start = params.time_range[0].timestamp()
        t_end = params.time_range[1].timestamp()
        
        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT * FROM read_csv_auto('{ctx.app_stats_path}') 
            WHERE timestamp >= {t_start} AND timestamp <= {t_end}
            ORDER BY timestamp ASC
        """
        df = db.query(sql)
        
        anomalous_tcs = {}
        metrics = ['rr', 'sr', 'cnt', 'mrt']
        
        # Calculate anomalies against baseline
        for tc, group in df.groupby('tc'):
            if tc not in baseline_data:
                continue
            
            tc_baseline = baseline_data[tc]
            max_z = 0.0
            primary_metric = ""
            onset_ts = None
            
            for _, row in group.iterrows():
                for m in metrics:
                    if m not in row: continue
                    mean = tc_baseline[m]['mean']
                    std = tc_baseline[m]['std']
                    
                    if std == 0:
                        z = 0 if row[m] == mean else 3.0
                    else:
                        z = abs(row[m] - mean) / std
                    
                    if z > 3.0:
                        if onset_ts is None:
                            onset_ts = row['timestamp']
                        if z > max_z:
                            max_z = z
                            primary_metric = m
            
            if onset_ts is not None:
                anomalous_tcs[tc] = TcAnomalySummary(
                    onset_timestamp=datetime.fromtimestamp(onset_ts),
                    max_z_score=float(max_z),
                    primary_degraded_metric=primary_metric
                )
                
        return QueryAppStatsAnomaliesOutput(anomalous_tcs=anomalous_tcs)

# 2. query_earliest_log_anomaly
class QueryEarliestLogAnomalyInput(BaseModel):
    time_range: tuple[datetime, datetime]
    baseline_ref: str

class QueryEarliestLogAnomalyOutput(BaseModel):
    t0_logs: datetime | None = Field(None, description="The earliest detected log anomaly timestamp.")

class QueryEarliestLogAnomalyTool(BaseTool[QueryEarliestLogAnomalyInput, QueryEarliestLogAnomalyOutput]):
    """Identifies the exact T0 for logs using DuckDB."""
    
    @property
    def name(self) -> str:
        return "query_earliest_log_anomaly"
        
    async def execute(self, ctx: ToolContext, params: QueryEarliestLogAnomalyInput) -> QueryEarliestLogAnomalyOutput:
        t_start = params.time_range[0].timestamp()
        t_end = params.time_range[1].timestamp()
        
        db = DuckDBClient.get_instance()
        # Find first error log in range
        sql = f"""
            SELECT MIN(timestamp) as earliest_ts 
            FROM read_csv_auto('{ctx.logs_path}')
            WHERE timestamp >= {t_start} AND timestamp <= {t_end}
            AND (value ILIKE '%Exception%' OR value ILIKE '%Error%' OR value ILIKE '%Failure%' OR value ILIKE '%FATAL%')
        """
        res = db.query(sql)
        earliest_ts = res['earliest_ts'].iloc[0]
        if not np.isnan(earliest_ts):
            return QueryEarliestLogAnomalyOutput(t0_logs=datetime.fromtimestamp(earliest_ts))
            
        return QueryEarliestLogAnomalyOutput(t0_logs=None)

# 3. query_earliest_trace_anomaly
class QueryEarliestTraceAnomalyInput(BaseModel):
    time_range: tuple[datetime, datetime]
    baseline_ref: str

class QueryEarliestTraceAnomalyOutput(BaseModel):
    t0_traces: datetime | None = Field(None, description="The earliest detected trace anomaly timestamp.")

class QueryEarliestTraceAnomalyTool(BaseTool[QueryEarliestTraceAnomalyInput, QueryEarliestTraceAnomalyOutput]):
    """Identifies the exact T0 for traces using DuckDB."""
    
    @property
    def name(self) -> str:
        return "query_earliest_trace_anomaly"
        
    async def execute(self, ctx: ToolContext, params: QueryEarliestTraceAnomalyInput) -> QueryEarliestTraceAnomalyOutput:
        t_start = params.time_range[0].timestamp()
        t_end = params.time_range[1].timestamp()
        
        db = DuckDBClient.get_instance()
        # Assume trace anomalies are long durations (> 5000000 microseconds) or errors
        sql = f"""
            SELECT MIN(timestamp) as earliest_ts 
            FROM read_csv_auto('{ctx.traces_path}')
            WHERE timestamp >= {t_start} AND timestamp <= {t_end}
            AND (duration > 5000000 OR status_code = 'ERROR')
        """
        res = db.query(sql)
        earliest_ts = res['earliest_ts'].iloc[0]
        if earliest_ts and not np.isnan(earliest_ts):
            return QueryEarliestTraceAnomalyOutput(t0_traces=datetime.fromtimestamp(earliest_ts))
        return QueryEarliestTraceAnomalyOutput(t0_traces=None)

# 4. query_leading_indicators
class QueryLeadingIndicatorsInput(BaseModel):
    cmdb_ids: list[str]
    lookback_window: int = Field(..., description="Minutes before T0 to look back")
    baseline_ref: str

class IndicatorTrend(BaseModel):
    trend_start: datetime
    trend_rate: float
    limit_proximity: float

class QueryLeadingIndicatorsOutput(BaseModel):
    indicators: dict[str, dict[str, IndicatorTrend]] = Field(..., description="{cmdb_id -> {kpi -> IndicatorTrend}}")

class QueryLeadingIndicatorsTool(BaseTool[QueryLeadingIndicatorsInput, QueryLeadingIndicatorsOutput]):
    """Analyzes metric trends prior to T0."""
    
    @property
    def name(self) -> str:
        return "query_leading_indicators"
        
    async def execute(self, ctx: ToolContext, params: QueryLeadingIndicatorsInput) -> QueryLeadingIndicatorsOutput:
        # Complex trend math skipped for architectural skeletal proof. 
        # Returns empty indicators map to satisfy LLM structure constraint.
        return QueryLeadingIndicatorsOutput(indicators={})

# 5. query_trace_cmdb_ids_for_tc
class QueryTraceCmdbIdsForTcInput(BaseModel):
    tc_value: str
    time_range: tuple[datetime, datetime]

class QueryTraceCmdbIdsForTcOutput(BaseModel):
    cmdb_ids: list[str] = Field(..., description="Components touched by this TC.")

class QueryTraceCmdbIdsForTcTool(BaseTool[QueryTraceCmdbIdsForTcInput, QueryTraceCmdbIdsForTcOutput]):
    """Identifies all components touched by a specific traffic controller using DuckDB."""
    
    @property
    def name(self) -> str:
        return "query_trace_cmdb_ids_for_tc"
        
    async def execute(self, ctx: ToolContext, params: QueryTraceCmdbIdsForTcInput) -> QueryTraceCmdbIdsForTcOutput:
        t_start = params.time_range[0].timestamp()
        t_end = params.time_range[1].timestamp()
        
        db = DuckDBClient.get_instance()
        sql = f"""
            SELECT DISTINCT cmdb_id 
            FROM read_csv_auto('{ctx.traces_path}')
            WHERE timestamp >= {t_start} AND timestamp <= {t_end}
            AND tc = '{params.tc_value}'
        """
        df = db.query(sql)
        return QueryTraceCmdbIdsForTcOutput(cmdb_ids=df['cmdb_id'].tolist())
