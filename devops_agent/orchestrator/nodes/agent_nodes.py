import asyncio
import json
import os
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from devops_agent.core.db.duckdb_client import DuckDBClient
from devops_agent.core.llm_provider import LLMFactory
from devops_agent.core.recovery.loop_prevention.stage_progress_tracker import StageProgressTracker
from devops_agent.tools.executor import ToolExecutor
from devops_agent.tools.interfaces.baseline_tools import (
    ComputeBaselineInput,
    ComputeBaselineStatisticsTool,
)
from devops_agent.tools.langchain_adapter import wrap_tools
from devops_agent.tools.models import ToolContext
from devops_agent.tools.registry import get_registry

from ..agents.context_assembler import ContextAssemblerService
from ..agents.schemas import RCAAgentOutput, TriageAgentOutput
from ..agents.triage_agent import TriageAgent
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
                categories = {}
                for k in cid_kpis:
                    cat = classify_kpi(k)
                    categories[cat] = categories.get(cat, 0) + 1
                    
                kpi_map[cid] = {
                    "available_categories": categories
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
        all_declared_components = set(declared_graph.keys())
        active_components = set(cmdb_ids) | set(tc_values) | set(log_cmdb_ids) | set(trace_cmdb_ids)
        healthy_components = list(all_declared_components - active_components)
            
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
    
    return parsed

class SubmitTriageReport(TriageAgentOutput):
    """Submit the final investigation brief and triage findings. Call this ONLY when you have finished using the other tools to investigate the data."""
    pass

class SubmitRCAReport(RCAAgentOutput):
    """Submit the final investigation report and RCA findings. Call this ONLY when you have isolated the root cause or hit an ambiguous/inconclusive state."""
    pass

async def triage_agent_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    agent = TriageAgent()
    bundle = agent.construct_prompt_bundle(state)
    
    registry = get_registry()
    executor = ToolExecutor(registry)
    inv_id_str = state.get("investigation_id", str(uuid.uuid4()))
    ctx = ToolContext(
        investigation_id=uuid.UUID(inv_id_str), 
        cluster_id="triage", 
        baseline_ref=state.get("baseline_registry_ref"),
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path"),
        topology_graph=state.get("declared_topology_graph", {})
    )
    
    # Tools needed for triage (Stages 1-4)
    t1 = registry.get("triage_query_metrics")
    t2 = registry.get("triage_query_app_stats")
    t3 = registry.get("triage_query_logs")
    t4 = registry.get("triage_query_traces")
    t5 = registry.get("run_connected_component_analysis")
    
    tools = wrap_tools([t1, t2, t3, t4, t5], executor, ctx)
    llm = LLMFactory.get_llm("triage").bind_tools(tools + [SubmitTriageReport])
    
    triage_system_prompt = """ROLE
--------
You are the Triage Agent. Your task is to investigate and determine the incident blast radius using diagnostic tools.

CONTEXT: DATA SOURCES & SCHEMAS
--------
1. Metrics (triage_query_metrics): 'timestamp', 'cmdb_id', 'kpi_name', 'value'
2. App Stats (triage_query_app_stats): 'timestamp', 'rr', 'sr', 'cnt', 'mrt', 'tc'. NO 'cmdb_id' or 'kpi_name'. Use 'tc' as component.
3. Logs (triage_query_logs): 'log_id', 'timestamp', 'cmdb_id', 'log_name', 'value'. NO 'kpi_name'.
4. Traces (triage_query_traces): 'timestamp', 'cmdb_id', 'parent_id', 'span_id', 'trace_id', 'duration'. NO 'kpi_name'.

CONSTRAINTS & RULES
--------
1. NO HALLUCINATION: Never assume a column exists if it is not explicitly listed in the schema for that specific source.
2. NO GUESSING: If a requested analysis requires columns that do not exist, use a different tool or source.
3. When finished, you MUST call SubmitTriageReport."""
    
    messages = [
        SystemMessage(content=triage_system_prompt),
        HumanMessage(content=f"Analyze the incident blast radius based on state:\n{json.dumps(bundle, default=str)}")
    ]
    
    tool_map = {t.name: t for t in tools}
    tracker = StageProgressTracker()
    
    while True:
        response = None
        for attempt in range(3):
            try:
                response = await llm.ainvoke(messages, config=config)
                break
            except Exception as e:
                if attempt == 2: raise e
                await asyncio.sleep(1 * (2 ** attempt))
        messages.append(response)
        
        if not response.tool_calls:
            messages.append(HumanMessage(content="You did not call any tools. You must call SubmitTriageReport to finish."))
            continue
            
        final_output = None
        for tc in response.tool_calls:
            name = tc["name"]
            if name == "SubmitTriageReport":
                final_output = tc["args"]
                break
            elif name in tool_map:
                tool = tool_map[name]
                tracker.record_tool_call(name, tc["args"])
                if tracker.is_looping():
                    final_output = {
                        "blast_radius": "localized",
                        "blast_radius_qualifier": "simultaneous",
                        "symptom_pattern": "Loop prevention triggered.",
                        "affected_component_candidates": [],
                        "investigation_cluster": [],
                        "ranked_hypotheses": [],
                        "investigation_state": "AMBIGUOUS_PRE_EVIDENCE",
                        "current_node": "triage"
                    }
                    break
                try:
                    result = await tool.ainvoke(tc["args"], config=config)
                except KeyError as e:
                    result = f"KeyError: {e}. Check your schema constraints. This column does not exist in the requested data source."
                except Exception as e:
                    result = f"Error executing tool: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            else:
                messages.append(ToolMessage(content=f"Error: Unknown tool {name}", tool_call_id=tc["id"]))
                
        if final_output is not None:
            return agent.parse_output(final_output)

async def rca_agent_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    from langchain_core.messages import ToolMessage

    from ..agents.rca_agent import RCAAgent
    
    agent = RCAAgent()
    bundle = agent.construct_prompt_bundle(state)
    
    registry = get_registry()
    executor = ToolExecutor(registry)
    inv_id_str = state.get("investigation_id", str(uuid.uuid4()))
    ctx = ToolContext(
        investigation_id=uuid.UUID(inv_id_str), 
        cluster_id="rca", 
        baseline_ref=state.get("baseline_registry_ref"),
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path")
    )
    
    # Load all RCA tools (Stages 5-8)
    tool_names = [
        "query_anomalous_traces", "build_span_tree_summary", 
        "query_metrics_for_hypothesis", "query_logs_for_hypothesis", 
        "extract_trace_dependency_edges", "run_propagation_direction_check", 
        "infer_metric_dependency_edges", "query_app_stats_detailed",
        "compute_metric_latency_correlation"
    ]
    
    available_tools = []
    for name in tool_names:
        try:
            available_tools.append(registry.get(name))
        except Exception:
            pass # Skip if not registered yet
            
    tools = wrap_tools(available_tools, executor, ctx)
    llm = LLMFactory.get_llm("rca").bind_tools(tools + [SubmitRCAReport])
    
    rca_system_prompt = """ROLE
--------
You are the RCA Agent. Your task is to deduce the root cause by gathering evidence for hypotheses.

CONTEXT: DATA SOURCES & SCHEMAS
--------
1. Metrics: 'timestamp', 'cmdb_id', 'kpi_name', 'value'
2. App Stats: NO 'cmdb_id' or 'kpi_name'. Use 'tc'.
3. Logs: NO 'kpi_name'. Use 'log_name'.
4. Traces: NO 'kpi_name'.

CONSTRAINTS & RULES
--------
1. NO HALLUCINATION: Never assume a column exists if it is not explicitly listed.
2. When finished, you MUST call SubmitRCAReport."""
    
    messages = [
        SystemMessage(content=rca_system_prompt),
        HumanMessage(content=f"Deduce root cause based on state:\n{json.dumps(bundle, default=str)}")
    ]
    
    tool_map = {t.name: t for t in tools}
    tracker = StageProgressTracker()
    
    while True:
        response = None
        for attempt in range(3):
            try:
                response = await llm.ainvoke(messages, config=config)
                break
            except Exception as e:
                if attempt == 2: raise e
                await asyncio.sleep(1 * (2 ** attempt))
        messages.append(response)
        
        if not response.tool_calls:
            messages.append(HumanMessage(content="You did not call any tools. You must call SubmitRCAReport to finish."))
            continue
            
        final_output = None
        for tc in response.tool_calls:
            name = tc["name"]
            if name == "SubmitRCAReport":
                final_output = tc["args"]
                break
            elif name in tool_map:
                tool = tool_map[name]
                tracker.record_tool_call(name, tc["args"])
                if tracker.is_looping():
                    final_output = {
                        "evidence_matrix": {},
                        "updated_hypothesis_scores": state.get("updated_hypothesis_scores", {}),
                        "eliminated_hypotheses": [],
                        "surviving_hypotheses": [],
                        "refined_dependency_graph": {},
                        "undeclared_dependencies": [],
                        "propagation_verified_pairs": [],
                        "investigation_state": "AMBIGUOUS",
                        "current_node": "rca",
                        "investigation_gaps": [{"reason": "Loop prevention triggered in RCA"}]
                    }
                    break
                try:
                    result = await tool.ainvoke(tc["args"], config=config)
                except KeyError as e:
                    result = f"KeyError: {e}. Check your schema constraints. This column does not exist in the requested data source."
                except Exception as e:
                    result = f"Error executing tool: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            else:
                messages.append(ToolMessage(content=f"Error: Unknown tool {name}", tool_call_id=tc["id"]))
                
        if final_output is not None:
            parsed = agent.parse_output(final_output)
            
            # Stage 5.5 Stopping Rule Check (Evidence-weighted probabilistic halting)
            from devops_agent.core.recovery.rca_convergence import RCAConvergenceEvaluator
            
            evaluator = RCAConvergenceEvaluator()
            signal = evaluator.evaluate(
                scores=parsed.get("updated_hypothesis_scores", {}),
                evidence_items=parsed.get("evidence_matrix", {}),
                tool_calls_made=tracker.total_calls,
                llm_confidence=parsed.get("confidence_level", "INCONCLUSIVE"),
                llm_investigation_state=parsed.get("investigation_state", "active"),
            )
            
            if signal.state_override is not None:
                parsed["investigation_state"] = signal.state_override
                if "investigation_gaps" not in parsed or not isinstance(parsed["investigation_gaps"], list):
                    parsed["investigation_gaps"] = []
                parsed["investigation_gaps"].append({
                    "reason": signal.override_reason,
                    "entropy": signal.hypothesis_entropy,
                    "gap": signal.top_hypothesis_gap,
                    "evidence_depth": signal.evidence_depth,
                })
            
            return parsed
