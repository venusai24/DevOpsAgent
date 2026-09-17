"""Deterministic control flow nodes."""

from typing import Any

from ..state import GLOBAL_FEEDBACK_KEY, InvestigationState


def reset_rca_branches_node(state: InvestigationState) -> dict[str, Any]:
    """Clear the per_branch_rca_results accumulator before a re-dispatch.

    ``per_branch_rca_results`` uses an append-only reducer so that concurrent
    fan-out branches can accumulate safely.  The custom ``_reset_or_extend``
    reducer treats an *empty* incoming list as a reset signal, so returning
    ``{"per_branch_rca_results": []}`` here wipes the accumulator without
    changing the field's Annotated type or touching any other state.
    """
    return {"per_branch_rca_results": []}


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
    try:
        from devops_agent.core.recovery.scoring_engine import DeterministicScorer
        from devops_agent.core.recovery.rca_convergence import RCAConvergenceEvaluator
        
        scorer = DeterministicScorer()
        
        hypotheses_raw = state.get("surviving_hypotheses", [])
        ranked = state.get("ranked_hypotheses", [])
        
        hypotheses = []
        for h in hypotheses_raw:
            if isinstance(h, str):
                hypotheses.append(h)
            elif isinstance(h, dict):
                name = h.get("name") or h.get("hypothesis") or h.get("id")
                if name:
                    hypotheses.append(name)
                    
        if not hypotheses and ranked:
            for h in ranked:
                if isinstance(h, dict):
                    name = h.get("name") or h.get("hypothesis") or h.get("id")
                    if name:
                        hypotheses.append(name)
                elif isinstance(h, str):
                    hypotheses.append(h)
                    
        if not hypotheses:
            hypotheses = ["hypothesis_1", "hypothesis_2"]
            
        evidence_items = state.get("evidence_items", [])
        evidence_log = state.get("evidence_log", [])
        critic_verdicts = state.get("critic_verdicts", [])
        
        prior_scores = state.get("updated_hypothesis_scores")
        if not prior_scores or not isinstance(prior_scores, dict):
            prior_scores = {}
            if ranked:
                for h in ranked:
                    if isinstance(h, dict):
                        name = h.get("name") or h.get("hypothesis") or h.get("id")
                        score = h.get("probability") or h.get("confidence") or h.get("score") or 0.0
                        if name:
                            prior_scores[name] = float(score)
        updated_scores, mismatched = scorer.score(
            hypotheses=hypotheses,
            evidence_items=evidence_items,
            evidence_log=evidence_log,
            critic_verdicts=critic_verdicts,
            prior_scores=prior_scores,
            topology_graph=state.get("discovered_topology_graph") or state.get("declared_topology_graph", {}),
            t0=state.get("T0"),
            investigation_cluster=state.get("investigation_cluster", []),
            root_cause_candidate=state.get("root_cause_candidate")
        )
        
        # Calculate mismatch ratio
        mismatch_ratio = 0.0
        if len(evidence_items) > 0:
            mismatch_ratio = len(mismatched) / len(evidence_items)
            
        threshold = scorer.weights.get("critic_trigger", {}).get("mismatch_ratio_threshold", 0.15)
        
        # Determine if we need to call the critic or correct deterministically
        unreviewed_mismatches = [m for m in mismatched if not any(cv.get("evidence_item_id") == m for cv in critic_verdicts)]
        needs_correction = False
        
        if mismatch_ratio > threshold and len(unreviewed_mismatches) > 0:
            needs_correction = True
            
        verification_failures = state.get("verification_failures", 0)

        review_items = [item for item in evidence_items if item.get("evidence_id", item.get("raw_reference", {}).get("evidence_id")) in unreviewed_mismatches]

        updates = {
            "updated_hypothesis_scores": updated_scores,
            "mismatched_items": mismatched,
            "current_evidence_items_to_review": review_items
        }

        if needs_correction:
            if verification_failures >= 3:
                # P4-6: Max loop cap — persistent correction failure escalates to HITL
                gaps = state.get("investigation_gaps", [])
                gaps.append({
                    "reason": f"Verification loop cap reached after {verification_failures} failures. Escalating to HITL.",
                    "mismatched_items": unreviewed_mismatches,
                })
                updates["verification_failures"] = verification_failures + 1
                updates["investigation_state"] = "AMBIGUOUS"
                updates["investigation_gaps"] = gaps
                updates["current_node"] = "deterministic_scoring_done"
            elif verification_failures == 0:
                # Attribute each unreviewed mismatch back to the branch that produced
                # it; anything with no matching evidence dict (e.g. a synthetic
                # topo_fail_* marker) falls into the global bucket, broadcast to
                # every branch on re-dispatch.
                evidence_id_to_branch = {
                    item.get("evidence_id"): item.get("branch_id") or GLOBAL_FEEDBACK_KEY
                    for item in review_items
                }
                feedback_by_branch: dict[str, list[str]] = {}
                for m in unreviewed_mismatches:
                    branch = evidence_id_to_branch.get(m, GLOBAL_FEEDBACK_KEY)
                    feedback_by_branch.setdefault(branch, []).append(f"- Evidence ID: {m}")
                feedback = {
                    branch: (
                        "CRITIC REJECTION (Deterministic): The following evidence items contradicted "
                        "the raw mathematical metrics, were temporally impossible, or topologically invalid. "
                        "Please re-evaluate:\n" + "\n".join(lines)
                    )
                    for branch, lines in feedback_by_branch.items()
                }
                updates["critic_feedback_for_rca"] = feedback
                updates["verification_failures"] = 1
                updates["current_node"] = "deterministic_scoring_needs_correction"
            else:
                updates["verification_failures"] = verification_failures + 1
                updates["current_node"] = "deterministic_scoring_needs_critic"
        else:
            updates["verification_failures"] = 0
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
            
            updates["current_node"] = "deterministic_scoring_done"
        
        return updates
    
    except Exception as e:
        # Verification engine safety wrapper - never fail silently
        gaps = state.get("investigation_gaps", [])
        gaps.append({
            "reason": f"Verification Engine Failure: {str(e)}"
        })
        return {
            "investigation_state": "AMBIGUOUS",
            "current_node": "deterministic_scoring_done",
            "investigation_gaps": gaps
        }
