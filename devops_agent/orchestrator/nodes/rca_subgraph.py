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
from ..agents.rca_agent import RCAAgent
from ..agents.schemas import SubmitEvidenceReport as SubmitEvidenceReportSchema

class SubmitEvidenceReport(SubmitEvidenceReportSchema):
    """Submit the evidence classification report. Call this ONLY when you have isolated the root cause or gathered sufficient evidence."""
    pass

class RCAState(InvestigationState):
    rca_messages: Annotated[list[AnyMessage], operator.add]
    rca_duplicates: int
    rca_fingerprints: list[str]
    rca_hypothesis_calls: dict[str, int]
    rca_hypothesis_scores: dict[str, list[float]]

def _global_stopping_condition_met(scores: dict[str, float], surviving: list[str]) -> bool:
    if not surviving:
        return True
    top_score = max([scores.get(h, 0.0) for h in surviving] + [0.0])
    if len(surviving) == 1 and top_score > 0.75:
        return True
    if top_score > 0.5 and len(surviving) == 0:
        return True
    return False

async def rca_llm_node(state: RCAState, config: RunnableConfig) -> dict[str, Any]:
    agent = RCAAgent()
    
    messages = state.get("rca_messages", [])
    if not messages:
        bundle = agent.construct_prompt_bundle(state)
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
2. NO NUMERICAL SCORING: You must ONLY classify evidence directionally (e.g. strongly_supports, neutral, weakly_contradicts). Do NOT attempt to calculate probabilities.
3. When finished, you MUST call SubmitEvidenceReport."""
        messages = [
            SystemMessage(content=rca_system_prompt),
            HumanMessage(content=f"Deduce root cause based on state:\n{json.dumps(bundle, default=str)}")
        ]
        
    registry = get_registry()
    executor = ToolExecutor(registry)
    inv_id_str = state.get("investigation_id", str(uuid.uuid4()))
    actual_graph = state.get("discovered_topology_graph") or state.get("declared_topology_graph", {})
    ctx = ToolContext(
        investigation_id=uuid.UUID(inv_id_str) if isinstance(inv_id_str, str) else inv_id_str, 
        cluster_id="rca", 
        baseline_ref=state.get("baseline_registry_ref"),
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path"),
        topology_graph=actual_graph
    )
    
    tool_names = [
        "query_anomalous_traces", "build_span_tree_summary", 
        "query_metrics_for_hypothesis", "query_logs_for_hypothesis", 
        "run_propagation_direction_check", "query_app_stats_detailed",
        "compute_metric_latency_correlation"
    ]
    
    available_tools = []
    for name in tool_names:
        try:
            available_tools.append(registry.get(name))
        except Exception:
            pass
            
    tools = wrap_tools(available_tools, executor, ctx)
    llm = LLMFactory.get_llm("rca").bind_tools(tools + [SubmitEvidenceReport])
    
    response = await llm.ainvoke(messages, config=config)
    return {"rca_messages": [response] if not state.get("rca_messages") else messages + [response]}

async def rca_tools_node(state: RCAState, config: RunnableConfig) -> dict[str, Any]:
    messages = state.get("rca_messages", [])
    last_msg = messages[-1]
    
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"rca_messages": [HumanMessage(content="You did not call any tools. You must call SubmitEvidenceReport to finish.")]}
        
    registry = get_registry()
    executor = ToolExecutor(registry)
    inv_id_str = state.get("investigation_id", str(uuid.uuid4()))
    actual_graph = state.get("discovered_topology_graph") or state.get("declared_topology_graph", {})
    ctx = ToolContext(
        investigation_id=uuid.UUID(inv_id_str) if isinstance(inv_id_str, str) else inv_id_str, 
        cluster_id="rca", 
        baseline_ref=state.get("baseline_registry_ref"),
        app_stats_path=state.get("app_stats_path"),
        metrics_path=state.get("metrics_path"),
        logs_path=state.get("logs_path"),
        traces_path=state.get("traces_path"),
        topology_graph=actual_graph
    )
    
    tool_names = [
        "query_anomalous_traces", "build_span_tree_summary", 
        "query_metrics_for_hypothesis", "query_logs_for_hypothesis", 
        "run_propagation_direction_check", "query_app_stats_detailed",
        "compute_metric_latency_correlation"
    ]
    available_tools = []
    for name in tool_names:
        try:
            available_tools.append(registry.get(name))
        except Exception:
            pass
    tools = wrap_tools(available_tools, executor, ctx)
    tool_map = {t.name: t for t in tools}
    
    new_messages = []
    
    rca_duplicates = state.get("rca_duplicates", 0)
    rca_fingerprints = state.get("rca_fingerprints", [])
    rca_hypothesis_calls = state.get("rca_hypothesis_calls", {})
    
    surviving = list(state.get("surviving_hypotheses", []))
    
    evidence_log = state.get("evidence_log", [])
    if not evidence_log:
        evidence_log = []
    else:
        evidence_log = list(evidence_log)
        
    eliminated_hypotheses = list(state.get("eliminated_hypotheses", []))
    
    for tc in last_msg.tool_calls:
        name = tc["name"]
        if name == "SubmitEvidenceReport":
            final_output = tc["args"]
            parsed = final_output.copy()
            parsed["current_node"] = "rca"
            parsed["evidence_log"] = evidence_log
            return parsed
            
        elif name in tool_map:
            tool = tool_map[name]
            
            # Loop prevention
            fingerprint = f"{name}:{json.dumps(tc['args'], sort_keys=True)}"
            if fingerprint in rca_fingerprints:
                rca_duplicates += 1
            else:
                rca_fingerprints = rca_fingerprints + [fingerprint]
                
            if rca_duplicates >= 3:
                # Force loop break
                parsed = {
                    "evidence_items": [],
                    "is_ready_to_conclude": False,
                    "eliminated_hypotheses": eliminated_hypotheses,
                    "surviving_hypotheses": surviving,
                    "refined_dependency_graph": {},
                    "undeclared_dependencies": [],
                    "propagation_verified_pairs": [],
                    "investigation_state": "AMBIGUOUS",
                    "current_node": "rca",
                    "investigation_gaps": [{"reason": "Loop prevention triggered in RCA"}],
                    "evidence_log": evidence_log
                }
                return parsed
                
            # Budget tracking
            current_hypothesis = tc["args"].get("hypothesis_name", surviving[0] if surviving else "mock_hyp")
            rca_hypothesis_calls[current_hypothesis] = rca_hypothesis_calls.get(current_hypothesis, 0) + 1
            
            # We skip advanced convergence logic here to keep it simple, but we enforce hard budget.
            total_calls = sum(rca_hypothesis_calls.values())
            if total_calls > 15 or rca_hypothesis_calls[current_hypothesis] > 5:
                # Budget exhausted
                parsed = {
                    "evidence_items": [],
                    "is_ready_to_conclude": False,
                    "eliminated_hypotheses": eliminated_hypotheses,
                    "surviving_hypotheses": surviving,
                    "refined_dependency_graph": {},
                    "undeclared_dependencies": [],
                    "propagation_verified_pairs": [],
                    "investigation_state": "AMBIGUOUS",
                    "current_node": "rca",
                    "investigation_gaps": [{"reason": "Budget exhausted in RCA"}],
                    "evidence_log": evidence_log
                }
                return parsed
                
            try:
                result = await tool.ainvoke(tc["args"], config=config)
                # Accumulate evidence
                if hasattr(result, "model_dump"):
                    evidence_log.append(result.model_dump())
                elif hasattr(result, "dict"):
                    evidence_log.append(result.dict())
                elif isinstance(result, dict):
                    evidence_log.append(result)
            except KeyError as e:
                result = f"KeyError: {e}. Check your schema constraints. This column does not exist in the requested data source."
            except Exception as e:
                result = f"Error executing tool: {e}"
            
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            new_messages.append(ToolMessage(content=f"Error: Unknown tool {name}", tool_call_id=tc["id"]))
            
    return {
        "rca_messages": new_messages,
        "rca_duplicates": rca_duplicates,
        "rca_fingerprints": rca_fingerprints,
        "rca_hypothesis_calls": rca_hypothesis_calls,
        "evidence_log": evidence_log,
        "surviving_hypotheses": surviving,
        "eliminated_hypotheses": eliminated_hypotheses
    }

def rca_should_continue(state: RCAState) -> str:
    messages = state.get("rca_messages", [])
    if not messages:
        return "rca_llm_node"
        
    last_msg = messages[-1]
    
    # Global stopping condition
    scores = state.get("updated_hypothesis_scores", {})
    surviving = state.get("surviving_hypotheses", [])
    if _global_stopping_condition_met(scores, surviving):
        return END
        
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "rca_tools_node"
        
    for tc in last_msg.tool_calls:
        if tc["name"] == "SubmitEvidenceReport":
            return END
            
    if state.get("rca_duplicates", 0) >= 3:
        return END
        
    rca_hypothesis_calls = state.get("rca_hypothesis_calls", {})
    if sum(rca_hypothesis_calls.values()) > 15:
        return END
        
    return "rca_tools_node"

def build_rca_subgraph():
    builder = StateGraph(RCAState)
    builder.add_node("rca_llm_node", rca_llm_node)
    builder.add_node("rca_tools_node", rca_tools_node)
    
    builder.set_entry_point("rca_llm_node")
    
    builder.add_conditional_edges(
        "rca_llm_node",
        rca_should_continue,
        {
            "rca_tools_node": "rca_tools_node",
            END: END
        }
    )
    
    builder.add_edge("rca_tools_node", "rca_llm_node")
    
    return builder.compile()
