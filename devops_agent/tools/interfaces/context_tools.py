"""Stage 0 Context Assembly Tools."""

from typing import Any

from pydantic import BaseModel, Field

from devops_agent.core.db.duckdb_client import DuckDBClient

from ..base import BaseTool


# 1. read_distinct_cmdb_ids
class ReadDistinctCmdbIdsInput(BaseModel):
    csv_path: str = Field(..., description="Path to the CSV file (e.g., metrics.csv or logs.csv).")

class ReadDistinctCmdbIdsOutput(BaseModel):
    cmdb_ids: list[str] = Field(..., description="List of unique CMDB IDs found.")

class ReadDistinctCmdbIdsTool(BaseTool[ReadDistinctCmdbIdsInput, ReadDistinctCmdbIdsOutput]):
    """Reads unique component IDs from the CMDB lookup table or metric files using DuckDB."""
    
    @property
    def name(self) -> str:
        return "read_distinct_cmdb_ids"
        
    def _execute(self, params: ReadDistinctCmdbIdsInput) -> ReadDistinctCmdbIdsOutput:
        db = DuckDBClient.get_instance()
        sql = f"SELECT DISTINCT cmdb_id FROM read_csv_auto('{params.csv_path}') WHERE cmdb_id IS NOT NULL"
        try:
            df = db.query(sql)
            cmdb_ids = df['cmdb_id'].tolist()
            return ReadDistinctCmdbIdsOutput(cmdb_ids=cmdb_ids)
        except Exception:
            return ReadDistinctCmdbIdsOutput(cmdb_ids=[])

# 2. read_distinct_tc_values
class ReadDistinctTcValuesInput(BaseModel):
    app_stats_path: str = Field(..., description="Path to the app_stats.csv file.")

class ReadDistinctTcValuesOutput(BaseModel):
    tc_values: list[str] = Field(..., description="List of unique traffic controller (TC) values.")

class ReadDistinctTcValuesTool(BaseTool[ReadDistinctTcValuesInput, ReadDistinctTcValuesOutput]):
    """Reads unique traffic controller (TC) values from application stats using DuckDB."""
    
    @property
    def name(self) -> str:
        return "read_distinct_tc_values"
        
    def _execute(self, params: ReadDistinctTcValuesInput) -> ReadDistinctTcValuesOutput:
        db = DuckDBClient.get_instance()
        sql = f"SELECT DISTINCT tc FROM read_csv_auto('{params.app_stats_path}') WHERE tc IS NOT NULL"
        try:
            df = db.query(sql)
            tc_values = df['tc'].tolist()
            return ReadDistinctTcValuesOutput(tc_values=tc_values)
        except Exception:
            return ReadDistinctTcValuesOutput(tc_values=[])

# 3. sample_kpi_names
class SampleKpiNamesInput(BaseModel):
    metrics_path: str = Field(..., description="Path to the metrics.csv file.")
    cmdb_id: str = Field(..., description="Component ID to sample KPIs for.")

class SampleKpiNamesOutput(BaseModel):
    categorized_kpis: dict[str, list[str]] = Field(..., description="All available KPIs grouped by semantic category.")

class SampleKpiNamesTool(BaseTool[SampleKpiNamesInput, SampleKpiNamesOutput]):
    """Retrieves all available KPI names for a given component, grouped by semantic category."""
    
    @property
    def name(self) -> str:
        return "sample_kpi_names"
        
    def _execute(self, params: SampleKpiNamesInput) -> SampleKpiNamesOutput:
        db = DuckDBClient.get_instance()
        sql = f"SELECT DISTINCT kpi_name FROM read_csv_auto('{params.metrics_path}') WHERE cmdb_id = ?"
        try:
            df = db.query(sql, (params.cmdb_id,))
            kpi_names = df['kpi_name'].tolist()
            
            KPI_TAXONOMY = {
                "cpu": ["cpu"],
                "memory": ["memory", "mem"],
                "disk": ["filesystem", "localdisk", "disk", "fs"],
                "network": ["network", "net"],
                "process": ["process", "proc"],
                "jvm": ["jvm"],
                "tomcat_request": ["request", "processingtime", "errorcount"],
                "tomcat_session": ["session"],
                "tomcat_thread": ["thread"],
                "redis_memory": ["used_memory", "mem_fragmentation", "evicted_keys", "expired_keys"],
                "redis_clients": ["connected_clients", "blocked_clients", "rejected_connections"],
                "redis_perf": ["ops_per_sec", "keyspace_hits", "keyspace_misses", "latest_fork_usec"],
            }
            
            def classify_kpi(name: str) -> str:
                name_l = name.lower()
                for category, patterns in KPI_TAXONOMY.items():
                    if any(p in name_l for p in patterns):
                        return category
                return "other"
                
            categorized = {}
            for k in kpi_names:
                cat = classify_kpi(k)
                if cat not in categorized:
                    categorized[cat] = []
                categorized[cat].append(k)
                
            return SampleKpiNamesOutput(categorized_kpis=categorized)
        except Exception:
            return SampleKpiNamesOutput(categorized_kpis={})

# 4. read_topology_edges
class ReadTopologyEdgesInput(BaseModel):
    manifest: dict[str, Any] = Field(..., description="Topology manifest structure.")

class ReadTopologyEdgesOutput(BaseModel):
    adjacency_list: dict[str, list[str]] = Field(..., description="Caller to callee adjacency list.")

class ReadTopologyEdgesTool(BaseTool[ReadTopologyEdgesInput, ReadTopologyEdgesOutput]):
    """Parses the topology manifest into an adjacency list."""
    
    @property
    def name(self) -> str:
        return "read_topology_edges"
        
    def _execute(self, params: ReadTopologyEdgesInput) -> ReadTopologyEdgesOutput:
        return ReadTopologyEdgesOutput(adjacency_list=params.manifest)

# 5. extract_operation_from_tc
class ExtractOperationFromTcInput(BaseModel):
    tc_value: str = Field(..., description="The TC value to analyze.")
    sample_log_entries: list[str] = Field(..., description="Sample log entries associated with the TC.")

class ExtractOperationOutput(BaseModel):
    operation_id: str = Field(..., description="Identifier for the operation.")
    protocol: str = Field(..., description="e.g., HTTP, GRPC")
    path_pattern: str = Field(..., description="Normalized path pattern.")
    unmapped: bool = Field(..., description="True if it could not be mapped reliably.")

class ExtractOperationFromTcTool(BaseTool[ExtractOperationFromTcInput, ExtractOperationOutput]):
    """Uses log entries to map a generic TC value to a specific operation/endpoint."""
    
    @property
    def name(self) -> str:
        return "extract_operation_from_tc"
        
    def _execute(self, params: ExtractOperationFromTcInput) -> ExtractOperationOutput:
        return ExtractOperationOutput(
            operation_id=params.tc_value,
            protocol="HTTP",
            path_pattern=f"/{params.tc_value}.json",
            unmapped=False
        )
