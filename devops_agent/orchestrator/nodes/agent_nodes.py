import os
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from devops_agent.core.db.duckdb_client import DuckDBClient
from devops_agent.tools.interfaces.baseline_tools import (
    ComputeBaselineInput,
    ComputeBaselineStatisticsTool,
)
from devops_agent.tools.interfaces.tracing_tools import (
    ExtractTraceDependencyEdgesInput,
    ExtractTraceDependencyEdgesTool,
)
from devops_agent.tools.models import ToolContext

from ..agents.context_assembler import ContextAssemblerService
from ..state import InvestigationState


def _should_disable_mocks() -> bool:
    return os.environ.get("DISABLE_MOCKS", "false").lower() == "true"

async def context_assembler_agent_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    # Programmatically compute baseline
    baseline_tool = ComputeBaselineStatisticsTool()
    
    inv_id_str = state.get("investigation_id", "")
    try:
        inv_id = uuid.UUID(inv_id_str)
    except ValueError:
        inv_id = uuid.uuid4()
        
    ctx = ToolContext(
        investigation_id=inv_id,
        cluster_id="initial_cluster",
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path")
    )
    
    start_str = state["time_range"][0].isoformat()
    end_str = state["time_range"][1].isoformat()
    
    baseline_paths = [
        os.path.join(os.getcwd(), "incident_data", "baseline_app_metrics.csv"),
        os.path.join(os.getcwd(), "incident_data", "baseline_container_metrics.csv")
    ]
    
    baseline_input = ComputeBaselineInput(
        csv_paths=baseline_paths,
        time_range_start=start_str,
        time_range_end=end_str,
        force_recompute=False
    )
    
    baseline_output = await baseline_tool.execute(ctx, baseline_input)
    baseline_ref = baseline_output.baseline_ref

    bundle = {
        "role": "CONTEXT_ASSEMBLER",
        "inputs": {
            "time_range": state.get("time_range"),
            "computed_baseline_ref": baseline_ref
        }
    }
    
    # Programmatically fetch topology/context data before LLM execution
    cmdb_ids = []
    tc_values = []
    kpi_map = {}
    
    try:
        db = DuckDBClient.get_instance()
        
        time_filter = ""
        time_filter_traces = ""
        time_range = state.get("time_range")
        if time_range and len(time_range) == 2:
            start_epoch = int(time_range[0].timestamp())
            end_epoch = int(time_range[1].timestamp())
            time_filter = f"AND timestamp >= {start_epoch} AND timestamp <= {end_epoch}"
            time_filter_traces = f"AND timestamp >= {start_epoch * 1000} AND timestamp <= {end_epoch * 1000}"

        if state.get("metrics_path"):
            df_cmdb = db.query(f"SELECT DISTINCT cmdb_id FROM read_csv_auto('{state.get('metrics_path')}') WHERE cmdb_id IS NOT NULL {time_filter}")
            cmdb_ids = df_cmdb['cmdb_id'].tolist()
            
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
            
            df_kpi_all = db.query(f"SELECT cmdb_id, kpi_name, COUNT(*) as count FROM read_csv_auto('{state.get('metrics_path')}') WHERE cmdb_id IS NOT NULL {time_filter} GROUP BY cmdb_id, kpi_name")
            
            for cid in cmdb_ids:
                cid_kpis = df_kpi_all[df_kpi_all['cmdb_id'] == cid]['kpi_name'].tolist()
                categories: dict[str, int] = {}
                for k in cid_kpis:
                    cat = classify_kpi(k)
                    categories[cat] = categories.get(cat, 0) + 1

                kpi_map[cid] = {
                    # Exact verbatim names the LLM must use with query_metrics_for_hypothesis
                    "available_metrics": cid_kpis,
                    "available_categories": categories,
                }
                
        if state.get("app_stats_path"):
            df_tc = db.query(f"SELECT DISTINCT tc FROM read_csv_auto('{state.get('app_stats_path')}') WHERE tc IS NOT NULL {time_filter}")
            tc_values = df_tc['tc'].tolist()
            
        log_cmdb_ids = []
        if state.get("logs_path"):
            df_log_cmdb = db.query(f"SELECT DISTINCT cmdb_id FROM read_csv_auto('{state.get('logs_path')}') WHERE cmdb_id IS NOT NULL {time_filter}")
            log_cmdb_ids = df_log_cmdb['cmdb_id'].tolist()
            
        trace_cmdb_ids = []
        if state.get("traces_path"):
            df_trace_cmdb = db.query(f"SELECT DISTINCT cmdb_id FROM read_csv_auto('{state.get('traces_path')}') WHERE cmdb_id IS NOT NULL {time_filter_traces}")
            trace_cmdb_ids = df_trace_cmdb['cmdb_id'].tolist()
            
        # Integrate Negative Space Concept
        declared_graph = state.get("declared_topology_graph", {})
        
        explicit_symptoms = state.get("explicit_symptoms", {})
        if isinstance(explicit_symptoms, dict):
            dependencies_unknown = explicit_symptoms.get("dependencies_unknown", False)
        else:
            dependencies_unknown = getattr(explicit_symptoms, "dependencies_unknown", False)

        if dependencies_unknown:
            # Drop declared edges, keeping only nodes for isolated investigation
            declared_graph = {node: [] for node in declared_graph.keys()}

        all_declared_components = set(declared_graph.keys())
        active_components = set(cmdb_ids) | set(tc_values) | set(log_cmdb_ids) | set(trace_cmdb_ids)
        healthy_components = list(all_declared_components - active_components)
        
        discovered_topology = {}
        if dependencies_unknown:
            trace_tool = ExtractTraceDependencyEdgesTool()
            t_input = ExtractTraceDependencyEdgesInput(
                time_range_start=start_str,
                time_range_end=end_str,
                cmdb_ids=list(active_components)
            )
            try:
                t_output = await trace_tool.execute(ctx, t_input)
                for caller, callees in t_output.caller_callee_frequencies.items():
                    discovered_topology[caller] = list(callees.keys())
            except Exception as e:
                print(f"Failed to extract trace dependencies: {e}")
            
    except Exception as e:
        print(f"Error querying programmatic context: {e}")
            
    bundle["inputs"]["discovered_cmdb_ids"] = cmdb_ids
    bundle["inputs"]["discovered_tc_values"] = tc_values
    bundle["inputs"]["discovered_kpi_map"] = kpi_map
    if 'healthy_components' in locals():
        bundle["inputs"]["healthy_components_status"] = f"NORMAL (NO ANOMALY/DATA IN WINDOW): {healthy_components}"
    
    # Deterministic Service execution
    service = ContextAssemblerService()
    parsed = service.assemble(state, cmdb_ids, tc_values)
    parsed["baseline_registry_ref"] = baseline_ref

    # Build a flat component_kpi_map: cmdb_id -> list[exact metric names]
    # This is the authoritative metric registry for the RCA LLM — it must
    # use only names from this map when calling query_metrics_for_hypothesis.
    parsed["component_kpi_map"] = {
        cid: info["available_metrics"]
        for cid, info in kpi_map.items()
        if "available_metrics" in info
    }

    if dependencies_unknown and 'discovered_topology' in locals():
        parsed["discovered_topology_graph"] = discovered_topology

    return parsed


