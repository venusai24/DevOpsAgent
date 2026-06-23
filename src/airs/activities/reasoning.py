"""
Reasoning Activity — Module 1.10.

Temporal activity: Run one LangGraph reasoning hop.
The activity invokes the LangGraph state machine to produce the next
ExecutionIntent, updating the pursuit state in the process.

Temporal contract:
  - Deterministic output for same state (LLM temperature=0.1 + same context)
  - Returns (ExecutionIntent, updated InvestigationState) as tuple
  - Retry-safe: LLM calls are idempotent at the intent level
"""
from __future__ import annotations

import logging

from temporalio import activity

from airs.graph.state_graph import run_reasoning_hop
from airs.models.intents import ExecutionIntent
from airs.models.investigation import InvestigationState

log = logging.getLogger(__name__)


@activity.defn(name="run_reasoning_hop")
async def run_reasoning_hop_activity(
    state: InvestigationState,
) -> tuple[ExecutionIntent, InvestigationState]:
    """
    Execute one LangGraph reasoning hop.

    Runs the graph: calculate_missing_mass → route_decision → [terminal_node].
    The terminal node calls the LLM to produce a structured ExecutionIntent.

    Args:
        state: Current InvestigationState.

    Returns:
        Tuple of (ExecutionIntent, updated_InvestigationState).
        The updated state has refreshed pursuit_state.
    """
    activity.logger.info(
        "Running reasoning hop %d for investigation=%s",
        state.total_hop_count, state.investigation_id,
    )

    intent, updated_state = run_reasoning_hop(state)

    activity.logger.info(
        "Hop %d decision: action=%s reasoning=%s",
        state.total_hop_count,
        intent.action.value,
        intent.reasoning[:100] if intent.reasoning else "",
    )

    return intent, updated_state
