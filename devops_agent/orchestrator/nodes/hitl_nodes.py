"""Human-In-The-Loop interrupt nodes."""

from typing import Any

from langgraph.types import interrupt

from ..state import InvestigationState


def hitl_pre_evidence_node(state: InvestigationState) -> dict[str, Any]:
    """Interrupts when hypothesis generation fails to find strong candidates."""
    payload = {
        "trigger": "AMBIGUOUS_PRE_EVIDENCE",
        "investigation_id": state.get("investigation_id"),
        "top_hypotheses": state.get("ranked_hypotheses", [])
    }
    # Suspend graph and wait for external input
    resume_action = interrupt(payload)
    return {"hitl_resume_action": resume_action}

def hitl_ambiguous_evidence_node(state: InvestigationState) -> dict[str, Any]:
    """Interrupts at Stage 5.5 when evidence is inconclusive."""
    payload = {
        "trigger": "AMBIGUOUS_EVIDENCE",
        "investigation_id": state.get("investigation_id"),
        "surviving_hypotheses": state.get("surviving_hypotheses", [])
    }
    resume_action = interrupt(payload)
    return {"hitl_resume_action": resume_action}

def hitl_inconclusive_node(state: InvestigationState) -> dict[str, Any]:
    """Interrupts at Stage 7.1/7.3 when root cause is inconclusive."""
    payload = {
        "trigger": "INCONCLUSIVE_ROOT_CAUSE",
        "investigation_id": state.get("investigation_id"),
        "root_cause_candidate": state.get("root_cause_candidate")
    }
    resume_action = interrupt(payload)
    return {"hitl_resume_action": resume_action}

def resume_after_hitl_node(state: InvestigationState) -> dict[str, Any]:
    """Merges human input into InvestigationState and clears active interrupts."""
    # In a real implementation, this parses the human payload and applies it.
    # The routing edge will handle the actual branch.
    return {"investigation_state": "active"}
