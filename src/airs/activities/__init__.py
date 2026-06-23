"""AIRS Activities Package — public exports."""
from airs.activities.context_management import admit_evidence_candidate
from airs.activities.diagnosis import produce_diagnosis
from airs.activities.escalation import produce_escalation
from airs.activities.initialize import initialize_investigation
from airs.activities.mcp_tool import execute_mcp_tool
from airs.activities.playbook_retrieval import retrieve_playbook_context
from airs.activities.reasoning import run_reasoning_hop_activity
from airs.activities.verification import verify_epistemic_state

ALL_ACTIVITIES = [
    initialize_investigation,
    run_reasoning_hop_activity,
    execute_mcp_tool,
    admit_evidence_candidate,
    verify_epistemic_state,
    retrieve_playbook_context,
    produce_diagnosis,
    produce_escalation,
]

__all__ = [
    "initialize_investigation",
    "run_reasoning_hop_activity",
    "execute_mcp_tool",
    "admit_evidence_candidate",
    "verify_epistemic_state",
    "retrieve_playbook_context",
    "produce_diagnosis",
    "produce_escalation",
    "ALL_ACTIVITIES",
]
