"""Human-In-The-Loop interrupt nodes."""

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from ..state import InvestigationState


# ──────────────────────────────────────────────────────────────────────────────
# HITL interrupt nodes  (graph pauses BEFORE each of these via interrupt_before)
# ──────────────────────────────────────────────────────────────────────────────

def hitl_pre_evidence_node(state: InvestigationState) -> dict[str, Any]:
    """Interrupts when hypothesis generation fails to find strong candidates."""
    payload = {
        "trigger": "AMBIGUOUS_PRE_EVIDENCE",
        "description": "Triage could not isolate a clear blast radius. The agent needs direction.",
        "investigation_id": state.get("investigation_id"),
        "top_hypotheses": state.get("ranked_hypotheses", []),
        "investigation_gaps": state.get("investigation_gaps", []),
    }
    # Graph suspends here — run-incident.py will prompt the user and call
    # graph.ainvoke(Command(resume={...})) to provide the response.
    human_response = interrupt(payload)
    return {"hitl_resume_action": human_response.get("action"), "hitl_human_hint": human_response.get("hint", "")}


def hitl_ambiguous_evidence_node(state: InvestigationState) -> dict[str, Any]:
    """Interrupts at Stage 5.5 when evidence is inconclusive."""
    payload = {
        "trigger": "AMBIGUOUS_EVIDENCE",
        "description": "RCA gathered evidence but could not distinguish between competing hypotheses.",
        "investigation_id": state.get("investigation_id"),
        "surviving_hypotheses": state.get("surviving_hypotheses", []),
        "updated_scores": state.get("updated_hypothesis_scores", {}),
        "investigation_gaps": state.get("investigation_gaps", []),
    }
    human_response = interrupt(payload)
    return {"hitl_resume_action": human_response.get("action"), "hitl_human_hint": human_response.get("hint", "")}


def hitl_inconclusive_node(state: InvestigationState) -> dict[str, Any]:
    """Interrupts at Stage 7 when root cause cannot be confirmed."""
    payload = {
        "trigger": "INCONCLUSIVE_ROOT_CAUSE",
        "description": "Scoring produced an INCONCLUSIVE result. The agent cannot determine root cause from the data.",
        "investigation_id": state.get("investigation_id"),
        "root_cause_candidate": state.get("root_cause_candidate"),
        "confidence_level": state.get("confidence_level"),
        "investigation_gaps": state.get("investigation_gaps", []),
    }
    human_response = interrupt(payload)
    return {"hitl_resume_action": human_response.get("action"), "hitl_human_hint": human_response.get("hint", "")}


# ──────────────────────────────────────────────────────────────────────────────
# Resume node  (runs after every HITL interrupt, before routing)
# ──────────────────────────────────────────────────────────────────────────────

def resume_after_hitl_node(state: InvestigationState) -> dict[str, Any]:
    """
    Merges human input back into InvestigationState.

    The human hint is injected as a HumanMessage into rca_messages (and
    triage_messages if applicable) so the LLM treats it identically to
    data returned from a tool call — no special-casing needed anywhere.
    """
    hint = state.get("hitl_human_hint") or ""
    action = state.get("hitl_resume_action", "FORCE_CLOSE")

    updates: dict[str, Any] = {"investigation_state": "active"}

    if hint:
        # Wrap the hint as a message the LLM will naturally read on its next turn.
        supervisor_msg = HumanMessage(
            content=(
                f"[Human Supervisor Note]\n"
                f"Action decided: {action}\n"
                f"Hint: {hint}\n\n"
                f"Please use this context to continue the investigation. "
                f"Prioritise any specific components, logs, or hypotheses mentioned above."
            )
        )

        # Inject into RCA messages — rca_llm_node will pick this up on resume.
        existing_rca = list(state.get("rca_messages", []) or [])
        updates["rca_messages"] = existing_rca + [supervisor_msg]

        # Also inject into triage messages in case we're restarting triage.
        existing_triage = list(state.get("triage_messages", []) or [])
        updates["triage_messages"] = existing_triage + [supervisor_msg]

    return updates
