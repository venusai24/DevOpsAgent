from typing import Any

from langgraph.types import interrupt

from ..state import InvestigationState


def assess_ambiguity(state: InvestigationState) -> bool:
    """Deterministic structural check for ambiguity."""
    # Assuming SubmitEvidenceReport is embedded in state or we check state directly
    # The RCA subgraph currently dumps its fields into state directly (like surviving_hypotheses, root_cause_candidate)
    
    rc_candidate = state.get("root_cause_candidate")
    if not rc_candidate:
        return True
        
    confidence = state.get("confidence_level")
    if confidence == "LOW" or confidence == "INCONCLUSIVE":
        return True
        
    unresolved = state.get("unresolved_questions", [])
    if len(unresolved) >= 2:
        return True
        
    causal = state.get("causal_chain", [])
    if not causal:
        return True
        
    contradicting = state.get("contradicting_evidence", [])
    if contradicting and not causal:
        return True
        
    return False

def confidence_gate(state: InvestigationState) -> str:
    if assess_ambiguity(state):
        return "ambiguity_node"
    return "deterministic_scoring" # or report_delivery depending on old graph

def ambiguity_node(state: InvestigationState) -> dict[str, Any]:
    payload = {
        "reason": "Incident does not match known playbooks cleanly and natural-intelligence reasoning could not reach adequate confidence, OR the confidence was structurally unsupported.",
        "hypotheses_considered": state.get("alternative_hypotheses", []),
        "supporting_evidence": state.get("supporting_evidence", []),
        "contradicting_evidence": state.get("contradicting_evidence", []),
        "unresolved_questions": state.get("unresolved_questions", []),
        "investigation_state": "AMBIGUOUS",
    }
    
    # Graph pauses here until Command(resume=...) is provided
    human_response = interrupt(payload)
    
    action = human_response.get("action")
    if action == "manual_root_cause":
        return {
            "root_cause_candidate": {"description": human_response.get("root_cause")},
            "confidence_level": "HIGH",
            "investigation_state": "active", # un-ambiguous it
            "current_node": "ambiguity_resolved"
        }
    if action == "hint":
        return {
            "human_hint": human_response.get("hint"),
            "investigation_state": "active",
            "current_node": "ambiguity_hinted"
        }
        
    return {"investigation_state": "complete", "current_node": "ambiguity_escalated"}
    
def route_after_ambiguity(state: InvestigationState) -> str:
    node = state.get("current_node")
    if node == "ambiguity_hinted":
        return "natural_intelligence_triage" # fallback to pure reasoning with the hint
    elif node == "ambiguity_resolved":
        return "deterministic_scoring" # proceed to end
    return "report_delivery" # escalated / force close
