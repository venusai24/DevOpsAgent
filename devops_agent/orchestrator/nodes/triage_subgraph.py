import json
import uuid
from typing import Any, Annotated, TypedDict
import operator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from devops_agent.core.llm_provider import LLMFactory
from devops_agent.tools.executor import ToolExecutor
from devops_agent.tools.langchain_adapter import wrap_tools
from devops_agent.tools.models import ToolContext
from devops_agent.tools.registry import get_registry

from ..state import InvestigationState
from ..agents.triage_agent import TriageAgent
from ..agents.schemas import TriageAgentOutput

class SubmitTriageReport(TriageAgentOutput):
    """Submit the final investigation brief and triage findings. Call this ONLY when you have finished using the other tools to investigate the data."""
    pass

class TriageState(InvestigationState):
    triage_messages: Annotated[list[AnyMessage], operator.add]
    triage_duplicates: int
    triage_fingerprints: list[str]

async def triage_llm_node(state: TriageState, config: RunnableConfig) -> dict[str, Any]:
    agent = TriageAgent()
    
    # If messages is empty, initialize it
    messages = state.get("triage_messages", [])
    if not messages:
        bundle = agent.construct_prompt_bundle(state)
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
3. DEPENDENCY GRAPH: The dependency graph has been automatically resolved for you. When calling run_connected_component_analysis, simply provide the anomalous components.
4. When finished, you MUST call SubmitTriageReport."""
        messages = [
            SystemMessage(content=triage_system_prompt),
            HumanMessage(content=f"Analyze the incident blast radius based on state:\n{json.dumps(bundle, default=str)}")
        ]
        
    registry = get_registry()
    executor = ToolExecutor(registry)
    inv_id_str = state.get("investigation_id", str(uuid.uuid4()))
    actual_graph = state.get("discovered_topology_graph") or state.get("declared_topology_graph", {})
    ctx = ToolContext(
        investigation_id=uuid.UUID(inv_id_str) if isinstance(inv_id_str, str) else inv_id_str, 
        cluster_id="triage", 
        baseline_ref=state.get("baseline_registry_ref"),
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path"),
        topology_graph=actual_graph
    )
    
    t1 = registry.get("triage_query_metrics")
    t2 = registry.get("triage_query_app_stats")
    t3 = registry.get("triage_query_logs")
    t4 = registry.get("triage_query_traces")
    t5 = registry.get("run_connected_component_analysis")
    
    tools = wrap_tools([t1, t2, t3, t4, t5], executor, ctx)
    llm = LLMFactory.get_llm("triage").bind_tools(tools + [SubmitTriageReport])
    
    response = await llm.ainvoke(messages, config=config)
    return {"triage_messages": [response] if not state.get("triage_messages") else messages + [response]}

async def triage_tools_node(state: TriageState, config: RunnableConfig) -> dict[str, Any]:
    messages = state.get("triage_messages", [])
    last_msg = messages[-1]
    
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"triage_messages": [HumanMessage(content="You did not call any tools. You must call SubmitTriageReport to finish.")]}
        
    registry = get_registry()
    executor = ToolExecutor(registry)
    inv_id_str = state.get("investigation_id", str(uuid.uuid4()))
    actual_graph = state.get("discovered_topology_graph") or state.get("declared_topology_graph", {})
    ctx = ToolContext(
        investigation_id=uuid.UUID(inv_id_str) if isinstance(inv_id_str, str) else inv_id_str, 
        cluster_id="triage", 
        baseline_ref=state.get("baseline_registry_ref"),
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path"),
        topology_graph=actual_graph
    )
    
    tools = wrap_tools([
        registry.get("triage_query_metrics"),
        registry.get("triage_query_app_stats"),
        registry.get("triage_query_logs"),
        registry.get("triage_query_traces"),
        registry.get("run_connected_component_analysis")
    ], executor, ctx)
    
    tool_map = {t.name: t for t in tools}
    
    new_messages = []
    
    triage_duplicates = state.get("triage_duplicates", 0)
    triage_fingerprints = state.get("triage_fingerprints", [])
    
    agent = TriageAgent()
    
    for tc in last_msg.tool_calls:
        name = tc["name"]
        if name == "SubmitTriageReport":
            final_output = tc["args"]
            parsed = agent.parse_output(final_output)
            # Add an empty state override for clean return mapping
            return parsed
            
        elif name in tool_map:
            tool = tool_map[name]
            
            # Loop prevention tracking
            fingerprint = f"{name}:{json.dumps(tc['args'], sort_keys=True)}"
            if fingerprint in triage_fingerprints:
                triage_duplicates += 1
            else:
                triage_fingerprints = triage_fingerprints + [fingerprint]
                
            if triage_duplicates >= 3:
                # Force loop break
                parsed = {
                    "blast_radius": "localized",
                    "blast_radius_qualifier": "simultaneous",
                    "symptom_pattern": "Loop prevention triggered.",
                    "affected_component_candidates": [],
                    "investigation_cluster": [],
                    "ranked_hypotheses": [],
                    "investigation_state": "AMBIGUOUS_PRE_EVIDENCE",
                    "current_node": "triage"
                }
                return parsed
                
            try:
                result = await tool.ainvoke(tc["args"], config=config)
            except KeyError as e:
                result = f"KeyError: {e}. Check your schema constraints. This column does not exist in the requested data source."
            except Exception as e:
                result = f"Error executing tool: {e}"
            
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            new_messages.append(ToolMessage(content=f"Error: Unknown tool {name}", tool_call_id=tc["id"]))
            
    return {
        "triage_messages": new_messages,
        "triage_duplicates": triage_duplicates,
        "triage_fingerprints": triage_fingerprints
    }

def triage_should_continue(state: TriageState) -> str:
    messages = state.get("triage_messages", [])
    if not messages:
        return "triage_llm_node"
        
    last_msg = messages[-1]
    
    # If LLM didn't call tools, route to tools so we can inject the warning message
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "triage_tools_node"
        
    for tc in last_msg.tool_calls:
        if tc["name"] == "SubmitTriageReport":
            return END
            
    # Loop break check
    if state.get("triage_duplicates", 0) >= 3:
        return END
        
    return "triage_tools_node"

def build_triage_subgraph():
    builder = StateGraph(TriageState)
    builder.add_node("triage_llm_node", triage_llm_node)
    builder.add_node("triage_tools_node", triage_tools_node)
    
    builder.set_entry_point("triage_llm_node")
    
    builder.add_conditional_edges(
        "triage_llm_node",
        triage_should_continue,
        {
            "triage_tools_node": "triage_tools_node",
            END: END
        }
    )
    
    builder.add_edge("triage_tools_node", "triage_llm_node")
    
    return builder.compile()
