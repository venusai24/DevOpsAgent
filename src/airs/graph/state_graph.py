"""
AIRS LangGraph State Graph — Module 1.9.

Defines the reasoning state machine used within each Temporal reasoning_activity.
The graph is invoked once per investigation hop and produces a single ExecutionIntent.

Graph Structure:
  START
    │
    ▼
  [calculate_missing_mass]   ← Pure function, updates pursuit state
    │
    ▼
  [route_decision]           ← Deterministic routing (no LLM)
    │
    ├─ execute_tool    ──► [build_tool_intent]     ← LLM: choose which tool + args
    │
    ├─ query_playbook  ──► [build_playbook_intent] ← LLM: formulate retrieval query
    │
    ├─ diagnose        ──► [build_diagnose_intent] ← LLM: confirm DIAGNOSE action
    │
    └─ escalate        ──► [build_escalate_intent] ← LLM: explain escalation reason
         │
         ▼
       END → returns ExecutionIntent

Each terminal node calls the LLM once (Stage B decision prompt) and returns
a validated ExecutionIntent. The LLM is NEVER called in routing — only in
the terminal nodes where we need a structured decision.

Note: The graph is compiled once and reused across hops (LangGraph compilation
is expensive and thread-safe after compilation).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from airs.config import settings
from airs.graph.nodes.calculate_missing_mass import compute_missing_mass
from airs.graph.nodes.route_decision import route_decision
from airs.graph.prompts.interpretation import (
    build_decide_next_action_prompt,
    build_diagnosis_prompt,
    build_escalation_prompt,
)
from airs.graph.prompts.tier1_constitution import get_constitution_prompt
from airs.models.intents import (
    ExecutionIntent,
    IntentAction,
    PlaybookQuery,
    SignalType,
    ToolSpec,
    ToolTier,
)
from airs.models.investigation import InvestigationState

log = logging.getLogger(__name__)

# ─── LangGraph State Schema ───────────────────────────────────────────────────
# LangGraph requires a TypedDict-like schema for the state passed through nodes.
# We wrap InvestigationState + output intent in a simple dict.

from typing import TypedDict


class GraphState(TypedDict):
    """State passed between LangGraph nodes."""
    investigation_state: InvestigationState
    intent: Optional[ExecutionIntent]
    route: Optional[str]
    error: Optional[str]


# ─── Graph Node Functions ─────────────────────────────────────────────────────

def node_calculate_missing_mass(state: GraphState) -> GraphState:
    """
    Node: compute updated pursuit state (missing mass).
    Pure function — no LLM call.
    """
    inv_state = state["investigation_state"]
    new_pursuit = compute_missing_mass(inv_state, epsilon=settings.epsilon_threshold)
    updated_inv = inv_state.model_copy(update={"pursuit_state": new_pursuit})
    return {**state, "investigation_state": updated_inv}


def node_route_decision(state: GraphState) -> GraphState:
    """
    Node: apply deterministic routing rules. Sets `route` field.
    """
    inv_state = state["investigation_state"]
    decision = route_decision(inv_state, max_hops=settings.max_hops)
    log.debug("Route decision at hop %d: %s", inv_state.total_hop_count, decision)
    return {**state, "route": decision}


def node_build_tool_intent(state: GraphState) -> GraphState:
    """
    Node: call LLM to decide WHICH tool to execute and with WHAT arguments.
    Returns ExecutionIntent(action=EXECUTE_TOOL).
    """
    inv_state = state["investigation_state"]
    try:
        intent = _call_llm_for_intent(inv_state)
        return {**state, "intent": intent}
    except Exception as e:
        log.error("node_build_tool_intent failed: %s", e)
        return {**state, "error": str(e), "intent": _fallback_intent(inv_state)}


def node_build_playbook_intent(state: GraphState) -> GraphState:
    """
    Node: call LLM to formulate a playbook retrieval query.
    Returns ExecutionIntent(action=QUERY_PLAYBOOK).
    """
    inv_state = state["investigation_state"]
    try:
        intent = _call_llm_for_intent(inv_state, force_action="QUERY_PLAYBOOK")
        return {**state, "intent": intent}
    except Exception as e:
        log.error("node_build_playbook_intent failed: %s", e)
        # Fallback: query diagnostic_knowledge with alert description
        return {
            **state,
            "error": str(e),
            "intent": ExecutionIntent(
                action=IntentAction.QUERY_PLAYBOOK,
                playbook_query=PlaybookQuery(
                    collection="diagnostic_knowledge",
                    query_text=inv_state.alert.description or inv_state.alert.alert_name,
                    k=3,
                ),
                reasoning="Fallback playbook query due to LLM error",
                missing_mass_at_decision=inv_state.pursuit_state.current_missing_mass,
            ),
        }


def node_build_diagnose_intent(state: GraphState) -> GraphState:
    """Node: emit DIAGNOSE intent."""
    inv_state = state["investigation_state"]
    return {
        **state,
        "intent": ExecutionIntent(
            action=IntentAction.DIAGNOSE,
            reasoning="Missing mass converged with high-confidence hypothesis",
            missing_mass_at_decision=inv_state.pursuit_state.current_missing_mass,
        ),
    }


def node_build_escalate_intent(state: GraphState) -> GraphState:
    """Node: emit ESCALATE intent."""
    inv_state = state["investigation_state"]
    risk = inv_state.risk_state

    reason = "Supermartingale alarm" if risk.should_escalate else "Max hops reached"
    return {
        **state,
        "intent": ExecutionIntent(
            action=IntentAction.ESCALATE,
            reasoning=f"{reason}. M_t={risk.supermartingale_value:.2f}, hops={inv_state.total_hop_count}",
            missing_mass_at_decision=inv_state.pursuit_state.current_missing_mass,
        ),
    }


def _route_on_decision(state: GraphState) -> str:
    """Conditional edge function: routes based on `route` field in state."""
    return state.get("route", "execute_tool")


# ─── LLM Caller ──────────────────────────────────────────────────────────────

def _call_llm_for_intent(
    inv_state: InvestigationState,
    force_action: Optional[str] = None,
) -> ExecutionIntent:
    """
    Call the primary LLM to produce a structured ExecutionIntent.
    Uses LiteLLM for model-agnostic invocation.
    """
    from litellm import completion

    # Build context window
    from airs.context.manager import ProactiveContextManager
    ctx_manager = ProactiveContextManager.from_settings()
    context_window = ctx_manager.get_context_window(inv_state)

    # Summarise hypotheses
    hypotheses_summary = _format_hypotheses(inv_state)

    # Available tools from registry
    from airs.mcp.registry import get_tool_registry
    registry = get_tool_registry()
    available_tools = [
        {
            "tool_name": t.tool_name,
            "mcp_server_id": t.mcp_server_id,
            "tier": t.tier.value,
            "signal_type": t.signal_type.value,
        }
        for t in registry.get_all_tools()
    ]

    risk = inv_state.risk_state
    pursuit = inv_state.pursuit_state
    alert = inv_state.alert

    user_prompt = build_decide_next_action_prompt(
        alert_name=alert.alert_name,
        service=alert.service or "unknown",
        namespace=alert.namespace or "default",
        hop_index=inv_state.total_hop_count,
        max_hops=settings.max_hops,
        missing_mass=pursuit.current_missing_mass,
        trajectory_risk=risk.trajectory_risk_score,
        supermartingale=risk.supermartingale_value,
        escalation_threshold=risk.escalation_threshold,
        lambda_threshold=risk.lambda_threshold,
        hypotheses_summary=hypotheses_summary,
        context_window=context_window,
        playbook_context="",  # Populated by playbook retrieval activity
        available_tools=available_tools,
        consecutive_low_delta=pursuit.consecutive_low_delta,
    )

    if force_action:
        user_prompt = f"Force action to {force_action}.\n\n" + user_prompt

    messages = [
        {"role": "system", "content": get_constitution_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=settings.primary_model,
        messages=messages,
        temperature=0.1,  # Low temperature for deterministic structured output
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or "{}"
    return _parse_intent_from_llm(raw_text, inv_state.pursuit_state.current_missing_mass)


def _parse_intent_from_llm(
    raw_text: str,
    missing_mass: float,
) -> ExecutionIntent:
    """Parse and validate the LLM's JSON output into an ExecutionIntent."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        log.error("LLM returned invalid JSON: %s\nRaw: %s", e, raw_text[:300])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    action_str = data.get("action", "EXECUTE_TOOL").upper()
    try:
        action = IntentAction(action_str)
    except ValueError:
        action = IntentAction.EXECUTE_TOOL

    tool_spec = None
    playbook_query = None

    if action == IntentAction.EXECUTE_TOOL:
        ts = data.get("tool_spec", {})
        tool_spec = ToolSpec(
            tool_name=ts.get("tool_name", "execute_query"),
            mcp_server_id=ts.get("mcp_server_id", "prometheus"),
            arguments=ts.get("arguments", {}),
            tier=ToolTier(ts.get("tier", 1)),
            signal_type=SignalType(ts.get("signal_type", "METRICS")),
            idempotent=ts.get("idempotent", True),
            expected_latency_seconds=ts.get("expected_latency_seconds", 10),
        )

    elif action == IntentAction.QUERY_PLAYBOOK:
        pq = data.get("playbook_query", {})
        playbook_query = PlaybookQuery(
            collection=pq.get("collection", "diagnostic_knowledge"),
            query_text=pq.get("query_text", ""),
            filters=pq.get("filters", {}),
            k=pq.get("k", 3),
        )

    return ExecutionIntent(
        action=action,
        tool_spec=tool_spec,
        playbook_query=playbook_query,
        reasoning=data.get("reasoning", ""),
        missing_mass_at_decision=missing_mass,
    )


def _fallback_intent(inv_state: InvestigationState) -> ExecutionIntent:
    """Emergency fallback intent when LLM call fails."""
    return ExecutionIntent(
        action=IntentAction.EXECUTE_TOOL,
        tool_spec=ToolSpec(
            tool_name="execute_query",
            mcp_server_id="prometheus",
            arguments={
                "query": f'rate(http_requests_total{{service="{inv_state.alert.service}"}}[5m])'
            },
            tier=ToolTier.OBSERVATION,
            signal_type=SignalType.METRICS,
        ),
        reasoning="Fallback to Prometheus error rate query due to LLM failure",
        missing_mass_at_decision=inv_state.pursuit_state.current_missing_mass,
    )


def _format_hypotheses(state: InvestigationState) -> str:
    """Format hypotheses for the LLM prompt."""
    if not state.hypotheses:
        return "No hypotheses established yet."
    lines = []
    for h in state.hypotheses[:3]:
        lines.append(
            f"  [{h.hypothesis_id[:8]}] conf={h.confidence:.2f} [{h.status.value}] "
            f"{h.statement}"
        )
    return "\n".join(lines)


# ─── Graph Compilation ────────────────────────────────────────────────────────

_compiled_graph: Optional[CompiledStateGraph] = None


def build_reasoning_graph() -> CompiledStateGraph:
    """
    Build and compile the AIRS reasoning state graph.
    Called once per worker process. Thread-safe after compilation.
    """
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("calculate_missing_mass", node_calculate_missing_mass)
    builder.add_node("route_decision", node_route_decision)
    builder.add_node("execute_tool", node_build_tool_intent)
    builder.add_node("query_playbook", node_build_playbook_intent)
    builder.add_node("diagnose", node_build_diagnose_intent)
    builder.add_node("escalate", node_build_escalate_intent)

    # Entry point
    builder.set_entry_point("calculate_missing_mass")

    # Linear edges
    builder.add_edge("calculate_missing_mass", "route_decision")

    # Conditional edge from route_decision → one of 4 terminal nodes
    builder.add_conditional_edges(
        "route_decision",
        _route_on_decision,
        {
            "execute_tool": "execute_tool",
            "query_playbook": "query_playbook",
            "diagnose": "diagnose",
            "escalate": "escalate",
        },
    )

    # All terminal nodes → END
    for terminal in ("execute_tool", "query_playbook", "diagnose", "escalate"):
        builder.add_edge(terminal, END)

    return builder.compile()


def get_reasoning_graph() -> CompiledStateGraph:
    """Return the compiled reasoning graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_reasoning_graph()
        log.info("Reasoning graph compiled successfully")
    return _compiled_graph


def run_reasoning_hop(
    investigation_state: InvestigationState,
) -> tuple[ExecutionIntent, InvestigationState]:
    """
    Run one reasoning hop through the LangGraph state machine.

    Args:
        investigation_state: The current full investigation state.

    Returns:
        (ExecutionIntent, updated_InvestigationState) — the decision and
        updated state (pursuit_state updated with new missing mass).
    """
    graph = get_reasoning_graph()

    initial_state: GraphState = {
        "investigation_state": investigation_state,
        "intent": None,
        "route": None,
        "error": None,
    }

    final_state = graph.invoke(initial_state)

    intent = final_state.get("intent")
    if intent is None:
        log.error("Reasoning graph returned no intent — using fallback")
        intent = _fallback_intent(investigation_state)

    # Return the updated investigation_state (with refreshed pursuit_state)
    updated_inv = final_state["investigation_state"]
    return intent, updated_inv
