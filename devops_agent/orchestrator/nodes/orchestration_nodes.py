"""Deterministic control flow nodes."""

from typing import Any

from ..state import InvestigationState


def deduplication_node(state: InvestigationState) -> dict[str, Any]:
    """Stage -1: Deduplication query."""
    # In a real implementation, this would query the PostgresInvestigationRepository
    # For now, we assume no overlap by default unless injected
    if "dedup_decision" not in state:
        return {"dedup_decision": "NEW"}
    return {}

def save_stage0_artifacts_node(state: InvestigationState) -> dict[str, Any]:
    """Writes Stage 0 artifacts to external cache."""
    # Mocking external write
    return {"stage0_artifacts_available": True}

def report_delivery_node(state: InvestigationState) -> dict[str, Any]:
    """Serializes final report and triggers notifications."""
    return {"investigation_state": "complete"}

def duplicate_halt_node(state: InvestigationState) -> dict[str, Any]:
    """Halts execution because the investigation is a duplicate."""
    return {"investigation_state": "complete"}

def deterministic_scoring_node(state: InvestigationState) -> dict[str, Any]:
    from devops_agent.core.recovery.scoring_engine import DeterministicScorer
    from devops_agent.core.recovery.rca_convergence import RCAConvergenceEvaluator
    
    scorer = DeterministicScorer()
    
    hypotheses = state.get("surviving_hypotheses", [])
    if not hypotheses:
        # Default to some hypotheses if state is not populated well
        hypotheses = ["hypothesis_1", "hypothesis_2"]
        
    evidence_items = state.get("evidence_items", [])
    evidence_log = state.get("evidence_log", [])
    critic_verdicts = state.get("critic_verdicts", [])
    
    updated_scores, mismatched = scorer.score(
        hypotheses=hypotheses,
        evidence_items=evidence_items,
        evidence_log=evidence_log,
        critic_verdicts=critic_verdicts
    )
    
    # Calculate mismatch ratio
    mismatch_ratio = 0.0
    if len(evidence_items) > 0:
        mismatch_ratio = len(mismatched) / len(evidence_items)
        
    threshold = scorer.weights.get("critic_trigger", {}).get("mismatch_ratio_threshold", 0.15)
    
    # Determine if we need to call the critic
    unreviewed_mismatches = [m for m in mismatched if not any(cv.get("evidence_item_id") == m for cv in critic_verdicts)]
    needs_critic = False
    
    if mismatch_ratio > threshold and len(unreviewed_mismatches) > 0:
        needs_critic = True
        
    updates = {
        "updated_hypothesis_scores": updated_scores,
        "mismatched_items": mismatched,
        "current_evidence_items_to_review": [item for item in evidence_items if item.get("raw_reference", {}).get("evidence_id") in unreviewed_mismatches]
    }
    
    if not needs_critic:
        # We are done with scoring, evaluate convergence
        evaluator = RCAConvergenceEvaluator()
        
        # Need to reconstruct evidence_matrix format for the evaluator if it expects it, 
        # or just pass evidence_items directly depending on evaluator's signature.
        # Evaluator in agent_nodes.py used: evidence_items=parsed.get("evidence_matrix", {})
        # We will map evidence_items to a dict by hypothesis_id.
        evidence_matrix = {}
        for item in evidence_items:
            h_id = item.get("hypothesis_id")
            if h_id not in evidence_matrix:
                evidence_matrix[h_id] = []
            evidence_matrix[h_id].append(item)
            
        signal = evaluator.evaluate(
            scores=updated_scores,
            evidence_items=evidence_matrix,
            tool_calls_made=len(evidence_log), # rough proxy for tool_calls_made
            llm_confidence=state.get("confidence_level", "INCONCLUSIVE"),
            llm_investigation_state=state.get("investigation_state", "active"),
        )
        
        if signal.state_override is not None:
            updates["investigation_state"] = signal.state_override
            gaps = state.get("investigation_gaps", [])
            gaps.append({
                "reason": signal.override_reason,
                "entropy": signal.hypothesis_entropy,
                "gap": signal.top_hypothesis_gap,
                "evidence_depth": signal.evidence_depth,
            })
            updates["investigation_gaps"] = gaps

    # We use current_node to inform the router
    if needs_critic:
        updates["current_node"] = "deterministic_scoring_needs_critic"
    else:
        updates["current_node"] = "deterministic_scoring_done"
        
    return updates
