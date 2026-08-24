import json
import operator
import uuid
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from devops_agent.core.llm_provider import LLMFactory
from devops_agent.tools.executor import ToolExecutor
from devops_agent.tools.langchain_adapter import wrap_tools
from devops_agent.tools.models import ToolContext
from devops_agent.tools.registry import get_registry

from ..agents.rca_agent import RCAAgent
from ..agents.schemas import SubmitEvidenceReport as SubmitEvidenceReportSchema
from ..state import InvestigationState


class SubmitEvidenceReport(SubmitEvidenceReportSchema):
    """Submit the evidence classification report. Call this ONLY when you have isolated the root cause or gathered sufficient evidence."""
    pass

class RCAState(InvestigationState):
    # Branch identity (set by dispatch_rca_fan_out)
    rca_target_component: str          # Which component this branch is scoped to
    rca_branch_id: str                 # Unique branch identifier (= component name)
    # Per-branch budget (set by dispatch_rca_fan_out, never shared across branches)
    rca_branch_total_budget: int       # Max total tool calls for this branch
    rca_branch_tool_count: int         # Running tool-call count for this branch
    rca_hyp_per_hypothesis_budget: int # Max tool calls per hypothesis
    # Loop-prevention state (branch-local)
    rca_duplicates: int
    rca_fingerprints: list[str]
    rca_hypothesis_calls: dict[str, int]
    rca_hypothesis_scores: dict[str, list[float]]
    rca_completed: bool

def _global_stopping_condition_met(scores: dict[str, float], surviving: list[str]) -> bool:
    if not scores:
        return False
    if not surviving:
        return True
    top_score = max([scores.get(h, 0.0) for h in surviving] + [0.0])
    if len(surviving) == 1 and top_score > 0.75:
        return True
    if top_score > 0.5 and len(surviving) == 0:
        return True
    return False

def _make_exhausted_branch_result(
    state: "RCAState",
    surviving: list[str],
    eliminated_hypotheses: list[dict],
    reason: str,
) -> dict:
    """Build a well-formed branch result dict when a branch exhausts its budget or hits loop prevention.

    The merge node expects the same shape as a normal SubmitEvidenceReport output;
    this helper ensures budget-exhausted branches produce a consistent, mergeable result.
    """
    return {
        "branch_id": state.get("rca_branch_id", "ALL"),
        "target_component": state.get("rca_target_component", "ALL"),
        "surviving_hypotheses": surviving,
        "eliminated_hypotheses": eliminated_hypotheses,
        "updated_hypothesis_scores": {},
        "evidence_items": [],
        "root_cause_candidate": None,
        "causal_chain": [],
        "primary_bottleneck": None,
        "refined_dependency_graph": {},
        "undeclared_dependencies": [],
        "propagation_verified_pairs": [],
        "investigation_state": "AMBIGUOUS",
        "confidence_level": "INCONCLUSIVE",
        "investigation_gaps": [{"reason": reason}],
        "narrative_summary": "",
        "final_report": None,
        "unconfirmed_links": [],
    }


async def rca_llm_node(state: RCAState, config: RunnableConfig) -> dict[str, Any]:
    agent = RCAAgent()

    messages = state.get("rca_messages", [])
    is_first_turn = not messages
    if is_first_turn:
        bundle = agent.construct_prompt_bundle(state)
        target_component = state.get("rca_target_component", "ALL")
        scope_note = (
            f"\n\nTARGET COMPONENT SCOPE: {target_component}\n"
            "Investigate ONLY evidence directly relevant to this component.\n"
            "You may query other components only when needed to verify propagation direction."
            if target_component != "ALL"
            else ""
        )
        rca_system_prompt = """ROLE
--------
You are the RCA Agent. Your task is to deduce the root cause by gathering evidence for hypotheses.

CONTEXT: DATA SOURCES & SCHEMAS
--------
1. Metrics: 'timestamp', 'cmdb_id', 'kpi_name', 'value'
2. App Stats: NO 'cmdb_id' or 'kpi_name'. Use 'tc'.
3. Logs: NO 'kpi_name'. Use 'log_name'.
4. Traces: NO 'kpi_name'.

CRITICAL CONSTRAINT: METRIC NAMES
--------
You must use exact kpi_name strings. Generic names like 'cpu', 'disk', 'memory', or 'network' WILL ALWAYS FAIL.
Examples of VALID names: 'Mysql-MySQL_3306_Select Scan', 'OSLinux-OSLinux_FILESYSTEM_-_FSUsedSpace'

Workflow for querying metrics:
  1. You MUST call list_available_metrics_for_component FIRST to get the exact valid names.
  2. Only then call query_metrics_for_hypothesis with the exact name.

CONSTRAINTS & RULES
--------
1. NO HALLUCINATION: Never assume a column exists if it is not explicitly listed.
2. NO NUMERICAL SCORING: You must ONLY classify evidence directionally (e.g. strongly_supports, neutral, weakly_contradicts). Do NOT attempt to calculate probabilities.
3. When finished, you MUST call SubmitEvidenceReport."""
        messages = [
            SystemMessage(content=rca_system_prompt),
            HumanMessage(
                content=(
                    f"Deduce root cause based on state:{scope_note}\n"
                    f"{json.dumps(bundle, default=str)}"
                )
            ),
        ]
        # Inject critic feedback if present from a previous RCA pass.
        # Read from the top-level state field (written by critic_agent_node).
        critic_feedback = state.get("critic_feedback_for_rca")
        if critic_feedback:
            messages.append(
                HumanMessage(
                    content=(
                        "[Critic Feedback from previous pass]\n"
                        f"{critic_feedback}\n\n"
                        "Please take this feedback into account as you investigate."
                    )
                )

    if not is_first_turn:
        has_counterfactual = any("Assume your current leading hypothesis is WRONG" in getattr(m, "content", "") for m in messages)
        if len(messages) >= 10 and not has_counterfactual:
            messages.append(HumanMessage(content="Assume your current leading hypothesis is WRONG. What evidence would disprove it? Query for that evidence now."))

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
        "list_available_metrics_for_component",
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

    # ── Message accumulation fix ────────────────────────────────────────────────
    # rca_messages uses operator.add (append-only reducer). Returning
    # `messages + [response]` on subsequent turns would re-append the entire
    # history, duplicating every message (and every tool_call_id), which causes
    # the 400 "Duplicate tool response" API error.
    #
    # Rule: first turn returns the full initial conversation (system + human +
    # response) so those messages are persisted into state. Every subsequent
    # turn returns only [response] — operator.add handles the append.
    if is_first_turn:
        return {"rca_messages": messages + [response]}
    return {"rca_messages": [response]}

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
        "list_available_metrics_for_component",
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
    if not surviving:
        surviving = [h.get("name") for h in state.get("ranked_hypotheses", []) if isinstance(h, dict) and h.get("name")]
    
    new_evidence = []
    
    eliminated_hypotheses = list(state.get("eliminated_hypotheses", []))
    
    # ── Per-branch budget state ──────────────────────────────────────────────
    branch_tool_count: int = state.get("rca_branch_tool_count", 0)
    branch_total_budget: int = state.get("rca_branch_total_budget", 10)
    hyp_budget: int = state.get("rca_hyp_per_hypothesis_budget", 4)
    branch_id: str = state.get("rca_branch_id", "ALL")

    for tc in last_msg.tool_calls:
        name = tc["name"]
        if name == "SubmitEvidenceReport":
            final_output = tc["args"]
            # Parse the evidence report
            parsed = final_output.copy()
            parsed.pop("evidence_log", None)
            parsed["rca_completed"] = True
            parsed["current_node"] = "rca"
            # Convert evidence_items from dicts/models to plain dicts
            raw_evidence = parsed.get("evidence_items", [])
            evidence_dicts = []
            for item in raw_evidence:
                if hasattr(item, "model_dump"):
                    evidence_dicts.append(item.model_dump())
                elif hasattr(item, "dict"):
                    evidence_dicts.append(item.dict())
                elif isinstance(item, dict):
                    evidence_dicts.append(item)
            parsed["evidence_items"] = evidence_dicts
            # Write to branch accumulator instead of clobbering shared state
            branch_result = {
                "branch_id": branch_id,
                "target_component": state.get("rca_target_component", "ALL"),
                "surviving_hypotheses": parsed.get("surviving_hypotheses", surviving),
                "eliminated_hypotheses": parsed.get("eliminated_hypotheses", eliminated_hypotheses),
                "updated_hypothesis_scores": {},  # Scoring done by deterministic_scoring_node
                "evidence_items": evidence_dicts,
                "root_cause_candidate": parsed.get("root_cause_candidate"),
                "causal_chain": parsed.get("causal_chain", []),
                "primary_bottleneck": parsed.get("primary_bottleneck"),
                "refined_dependency_graph": parsed.get("refined_dependency_graph", {}),
                "undeclared_dependencies": parsed.get("undeclared_dependencies", []),
                "propagation_verified_pairs": parsed.get("propagation_verified_pairs", []),
                "investigation_state": parsed.get("investigation_state", "active"),
                "confidence_level": parsed.get("confidence_level"),
                "investigation_gaps": parsed.get("investigation_gaps", []),
                "narrative_summary": parsed.get("narrative_summary", ""),
                "final_report": parsed.get("final_report"),
                "unconfirmed_links": parsed.get("unconfirmed_links", []),
            }
            return {
                "per_branch_rca_results": [branch_result],
                "evidence_log": evidence_dicts,
                "rca_branch_tool_count": branch_tool_count,
                "rca_completed": True,
            }

        elif name in tool_map:
            tool = tool_map[name]

            # Loop prevention check moved to after the loop
            fingerprint = f"{name}:{json.dumps(tc['args'], sort_keys=True)}"
            if fingerprint in rca_fingerprints:
                rca_duplicates += 1
            else:
                rca_fingerprints = rca_fingerprints + [fingerprint]

            # Per-branch budget counters
            current_hypothesis = (
                tc["args"].get("hypothesis_id") or
                tc["args"].get("hypothesis_name") or
                (surviving[0] if surviving else "mock_hyp")
            )
            rca_hypothesis_calls[current_hypothesis] = (
                rca_hypothesis_calls.get(current_hypothesis, 0) + 1
            )
            branch_tool_count += 1
                
            try:
                result = await tool.ainvoke(tc["args"], config=config)
                # Accumulate evidence
                if hasattr(result, "model_dump"):
                    new_evidence.append(result.model_dump())
                elif hasattr(result, "dict"):
                    new_evidence.append(result.dict())
                elif isinstance(result, dict):
                    new_evidence.append(result)
            except KeyError as e:
                result = f"KeyError: {e}. Check your schema constraints. This column does not exist in the requested data source."
            except Exception as e:
                result = f"Error executing tool: {e}"
            
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            new_messages.append(ToolMessage(content=f"Error: Unknown tool {name}", tool_call_id=tc["id"]))
            
    # ── Enforce Budgets & Loop Prevention ──
    if rca_duplicates >= 3:
        branch_result = _make_exhausted_branch_result(
            state, surviving, eliminated_hypotheses,
            reason="Loop prevention triggered in RCA",
        )
        return {
            "per_branch_rca_results": [branch_result],
            "rca_completed": True,
        }

    max_hyp_calls = max(rca_hypothesis_calls.values()) if rca_hypothesis_calls else 0
    if branch_tool_count > branch_total_budget or max_hyp_calls > hyp_budget:
        branch_result = _make_exhausted_branch_result(
            state, surviving, eliminated_hypotheses,
            reason=(
                f"Per-branch budget exhausted (branch={branch_id}, "
                f"total_calls={branch_tool_count}/{branch_total_budget}, "
                f"max_hyp_calls={max_hyp_calls}/{hyp_budget})"
            ),
        )
        return {
            "per_branch_rca_results": [branch_result],
            "rca_branch_tool_count": branch_tool_count,
            "rca_hypothesis_calls": rca_hypothesis_calls,
            "rca_completed": True,
        }

    return {
        "rca_messages": new_messages,
        "rca_duplicates": rca_duplicates,
        "rca_fingerprints": rca_fingerprints,
        "rca_hypothesis_calls": rca_hypothesis_calls,
        "rca_branch_tool_count": branch_tool_count,
        "evidence_log": new_evidence,
        "surviving_hypotheses": surviving,
        "eliminated_hypotheses": eliminated_hypotheses,
    }

def rca_should_continue(state: RCAState) -> str:
    messages = state.get("rca_messages", [])
    if not messages:
        return "rca_llm_node"

    last_msg = messages[-1]

    # Global stopping condition (score-based convergence)
    scores = state.get("updated_hypothesis_scores", {})
    surviving = state.get("surviving_hypotheses", [])
    if _global_stopping_condition_met(scores, surviving):
        return END

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "rca_tools_node"

    for tc in last_msg.tool_calls:
        if tc["name"] == "SubmitEvidenceReport":
            return "rca_tools_node"

    # Per-branch budget guard (avoids even entering the tools node when over budget)
    branch_tool_count = state.get("rca_branch_tool_count", 0)
    branch_total_budget = state.get("rca_branch_total_budget", 10)
    if state.get("rca_duplicates", 0) >= 3 or branch_tool_count > branch_total_budget:
        return "rca_tools_node"

    return "rca_tools_node"

def rca_tools_condition(state: RCAState) -> str:
    if state.get("rca_completed"):
        return END
    return "rca_llm_node"

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
    
    builder.add_conditional_edges(
        "rca_tools_node",
        rca_tools_condition,
        {
            END: END,
            "rca_llm_node": "rca_llm_node"
        }
    )
    
    return builder.compile()
