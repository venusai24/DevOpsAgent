"""LangGraph orchestrator construction (4-Agent System)."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .nodes.agent_nodes import (
    context_assembler_agent_node,
    rca_agent_node,
    triage_agent_node,
)
from .nodes.hitl_nodes import (
    hitl_ambiguous_evidence_node,
    hitl_inconclusive_node,
    hitl_pre_evidence_node,
    resume_after_hitl_node,
)
from .nodes.orchestration_nodes import (
    deduplication_node,
    duplicate_halt_node,
    report_delivery_node,
    save_stage0_artifacts_node,
)
from .state import InvestigationState


def route_after_deduplication(state: InvestigationState) -> str:
    decision = state.get("dedup_decision", "NEW")
    if decision == "DUPLICATE":
        return "duplicate_halt"
    elif decision == "SUBSET" and state.get("stage0_artifacts_available"):
        return "triage"
    else:
        return "context_assembly"

def route_after_triage(state: InvestigationState) -> str:
    inv_state = state.get("investigation_state", "active")
    if inv_state == "AMBIGUOUS_PRE_EVIDENCE":
        return "hitl_pre_evidence"
        
    affected = state.get("affected_component_candidates", [])
    if not affected or state.get("incident_state") == "resolved":
        return "report_delivery"
        
    return "rca"

def route_after_rca(state: InvestigationState) -> str:
    conf = state.get("confidence_level")
    if conf in ("HIGH", "MEDIUM", "LOW"):
        return "report_delivery"
    elif conf == "INCONCLUSIVE":
        return "hitl_inconclusive"
    # Also catches AMBIGUOUS from Stage 5.5
    if state.get("investigation_state") == "AMBIGUOUS":
        return "hitl_ambiguous_evidence"
    return "report_delivery"

def route_after_hitl_resume(state: InvestigationState) -> str:
    action = state.get("hitl_resume_action")
    if action == "RESTART_TRIAGE":
        return "triage"
    elif action == "RESTART_RCA":
        return "rca"
    elif action == "RESTART_CONTEXT":
        return "context_assembly"
    elif action == "FORCE_CLOSE":
        return "report_delivery"
    return "report_delivery"

def build_investigation_graph(checkpointer: BaseCheckpointSaver = None):
    builder = StateGraph(InvestigationState)
    
    # Register nodes
    builder.add_node("deduplication", deduplication_node)
    builder.add_node("context_assembly", context_assembler_agent_node)
    builder.add_node("save_stage0_artifacts", save_stage0_artifacts_node)
    builder.add_node("triage", triage_agent_node)
    builder.add_node("rca", rca_agent_node)
    builder.add_node("hitl_pre_evidence", hitl_pre_evidence_node)
    builder.add_node("hitl_ambiguous_evidence", hitl_ambiguous_evidence_node)
    builder.add_node("hitl_inconclusive", hitl_inconclusive_node)
    builder.add_node("resume_after_hitl", resume_after_hitl_node)
    builder.add_node("report_delivery", report_delivery_node)
    builder.add_node("duplicate_halt", duplicate_halt_node)
    
    # Entry point
    builder.set_entry_point("deduplication")
    
    # Edges
    builder.add_conditional_edges("deduplication", route_after_deduplication, {
        "context_assembly": "context_assembly",
        "triage": "triage",
        "duplicate_halt": "duplicate_halt",
    })
    
    builder.add_edge("context_assembly", "save_stage0_artifacts")
    builder.add_edge("save_stage0_artifacts", "triage")
    
    builder.add_conditional_edges("triage", route_after_triage, {
        "rca": "rca",
        "hitl_pre_evidence": "hitl_pre_evidence",
        "report_delivery": "report_delivery",
    })
    
    builder.add_conditional_edges("rca", route_after_rca, {
        "report_delivery": "report_delivery",
        "hitl_ambiguous_evidence": "hitl_ambiguous_evidence",
        "hitl_inconclusive": "hitl_inconclusive",
    })
    
    builder.add_edge("hitl_pre_evidence", "resume_after_hitl")
    builder.add_edge("hitl_ambiguous_evidence", "resume_after_hitl")
    builder.add_edge("hitl_inconclusive", "resume_after_hitl")
    
    builder.add_conditional_edges("resume_after_hitl", route_after_hitl_resume, {
        "triage": "triage",
        "rca": "rca",
        "context_assembly": "context_assembly",
        "report_delivery": "report_delivery",
    })
    
    builder.add_edge("report_delivery", END)
    builder.add_edge("duplicate_halt", END)
    
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "hitl_pre_evidence",
            "hitl_ambiguous_evidence",
            "hitl_inconclusive"
        ]
    )
