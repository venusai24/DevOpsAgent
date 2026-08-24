"""Deterministic RCA Results Merge Node.

After all parallel RCA branches complete, LangGraph routes to this node.
It reads ``per_branch_rca_results`` (the operator.add accumulator) and
applies a deterministic merge strategy to produce a single, unified view
of the investigation that the downstream scoring and critic nodes expect.

Merge strategies per field
──────────────────────────
surviving_hypotheses        → union across all branches, deduplicated
eliminated_hypotheses       → union (a hypothesis eliminated in any branch is kept eliminated)
updated_hypothesis_scores   → weighted average; evidence-depth-weighted per branch
evidence_items              → concatenated across all branches
root_cause_candidate        → branch with confidence_level == "HIGH" wins;
                              if multiple HIGH-confidence branches agree on the same
                              component, that agreement boosts the final confidence note;
                              tie-break by highest score
causal_chain                → from the winning root_cause_candidate branch
primary_bottleneck          → from the winning branch
refined_dependency_graph    → merged dict (later branches extend earlier ones)
undeclared_dependencies     → concatenated, deduplicated by (from_component, to_component)
propagation_verified_pairs  → concatenated, deduplicated by (source, target)
final_report                → assembled from winning branch + cross-branch summary block
investigation_state         → worst-case escalation
                              (ACTIVE < AMBIGUOUS < INCONCLUSIVE — any branch can escalate)
investigation_gaps          → union of all branches
confidence_level            → worst-case (INCONCLUSIVE < LOW < MEDIUM < HIGH)
"""

from __future__ import annotations

import logging
from typing import Any

from ..state import InvestigationState

logger = logging.getLogger(__name__)

# Confidence level ordering (lower index = lower confidence)
_CONFIDENCE_RANK: dict[str, int] = {
    "INCONCLUSIVE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}

# Investigation state ordering (lower index = worse)
_INV_STATE_RANK: dict[str, int] = {
    "INCONCLUSIVE": 0,
    "AMBIGUOUS": 1,
    "AMBIGUOUS_PRE_EVIDENCE": 2,
    "active": 3,
    "complete": 4,
}


def merge_rca_results_node(state: InvestigationState) -> dict[str, Any]:
    """Merge all parallel RCA branch results into the shared InvestigationState."""
    branches: list[dict[str, Any]] = state.get("per_branch_rca_results", [])

    if not branches:
        logger.warning("merge_rca_results_node: no branch results found — nothing to merge.")
        return {"current_node": "merge_rca"}

    logger.info(
        "merge_rca_results_node: merging %d branch(es): %s",
        len(branches),
        [b.get("target_component", "?") for b in branches],
    )

    # ── surviving_hypotheses ─────────────────────────────────────────────────
    surviving: list[str] = _deduplicated_union(
        [b.get("surviving_hypotheses", []) for b in branches]
    )

    # ── eliminated_hypotheses ────────────────────────────────────────────────
    eliminated_raw: list[dict] = []
    seen_elim: set[str] = set()
    for b in branches:
        for item in b.get("eliminated_hypotheses", []):
            key = item.get("name") or item.get("hypothesis") or str(item)
            if key not in seen_elim:
                seen_elim.add(key)
                eliminated_raw.append(item)
    # Remove any hypothesis from surviving if it was eliminated by any branch
    surviving = [h for h in surviving if h not in seen_elim]

    # ── updated_hypothesis_scores ────────────────────────────────────────────
    merged_scores = _merge_scores(branches)

    # ── evidence_items ───────────────────────────────────────────────────────
    all_evidence: list[dict] = []
    for b in branches:
        all_evidence.extend(b.get("evidence_items", []))

    # ── root_cause_candidate / winning branch ────────────────────────────────
    winning_branch = _pick_winning_branch(branches, merged_scores)

    root_cause_candidate = winning_branch.get("root_cause_candidate") if winning_branch else None
    causal_chain = winning_branch.get("causal_chain", []) if winning_branch else []
    primary_bottleneck = winning_branch.get("primary_bottleneck") if winning_branch else None

    # Cross-branch agreement bonus: if ≥2 branches point to the same root component
    if root_cause_candidate:
        rc_component = _extract_component_id(root_cause_candidate)
        agreeing = [
            b for b in branches
            if _extract_component_id(b.get("root_cause_candidate")) == rc_component
            and b.get("root_cause_candidate") is not None
        ]
        if len(agreeing) >= 2 and root_cause_candidate:
            root_cause_candidate = dict(root_cause_candidate)
            root_cause_candidate["cross_branch_agreement"] = len(agreeing)

    # ── refined_dependency_graph ─────────────────────────────────────────────
    merged_dep_graph: dict[str, list] = {}
    for b in branches:
        for node, edges in b.get("refined_dependency_graph", {}).items():
            if node not in merged_dep_graph:
                merged_dep_graph[node] = []
            for edge in edges:
                if edge not in merged_dep_graph[node]:
                    merged_dep_graph[node].append(edge)

    # ── undeclared_dependencies ──────────────────────────────────────────────
    undeclared = _deduplicate_pairs(
        [dep for b in branches for dep in b.get("undeclared_dependencies", [])],
        key_fields=("from_component", "to_component"),
    )

    # ── propagation_verified_pairs ───────────────────────────────────────────
    prop_pairs = _deduplicate_pairs(
        [p for b in branches for p in b.get("propagation_verified_pairs", [])],
        key_fields=("source", "target"),
    )

    # ── investigation_state — worst-case escalation ──────────────────────────
    merged_inv_state = _worst_case_investigation_state(branches)

    # ── confidence_level — worst-case ────────────────────────────────────────
    merged_confidence = _worst_case_confidence(branches)

    # ── investigation_gaps ───────────────────────────────────────────────────
    all_gaps: list[dict] = []
    for b in branches:
        all_gaps.extend(b.get("investigation_gaps", []))

    # ── final_report — winning branch + cross-branch summary ─────────────────
    final_report = _build_final_report(winning_branch, branches, root_cause_candidate)

    # ── narrative_summary ────────────────────────────────────────────────────
    narrative = winning_branch.get("narrative_summary", "") if winning_branch else ""

    updates: dict[str, Any] = {
        "surviving_hypotheses": surviving,
        "eliminated_hypotheses": eliminated_raw,
        "updated_hypothesis_scores": merged_scores,
        "evidence_items": all_evidence,
        "root_cause_candidate": root_cause_candidate,
        "causal_chain": causal_chain,
        "primary_bottleneck": primary_bottleneck,
        "refined_dependency_graph": merged_dep_graph,
        "undeclared_dependencies": undeclared,
        "propagation_verified_pairs": prop_pairs,
        "investigation_state": merged_inv_state,
        "confidence_level": merged_confidence,
        "investigation_gaps": all_gaps,
        "final_report": final_report,
        "current_node": "merge_rca",
    }

    if narrative:
        updates["narrative_summary"] = narrative

    logger.info(
        "merge_rca_results_node: merged → root_cause=%s confidence=%s state=%s",
        _extract_component_id(root_cause_candidate),
        merged_confidence,
        merged_inv_state,
    )

    return updates


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deduplicated_union(lists: list[list[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _merge_scores(branches: list[dict]) -> dict[str, float]:
    """Weighted-average hypothesis scores across branches.

    Weight is proportional to the number of evidence items each branch produced
    (evidence-depth weighting). Branches with no evidence get equal weight.
    """
    score_accum: dict[str, list[tuple[float, float]]] = {}  # hyp → [(score, weight)]

    for b in branches:
        scores: dict[str, float] = b.get("updated_hypothesis_scores", {})
        weight = float(max(len(b.get("evidence_items", [])), 1))
        for hyp, score in scores.items():
            if hyp not in score_accum:
                score_accum[hyp] = []
            score_accum[hyp].append((float(score), weight))

    merged: dict[str, float] = {}
    for hyp, weighted_scores in score_accum.items():
        total_weight = sum(w for _, w in weighted_scores)
        merged[hyp] = sum(s * w for s, w in weighted_scores) / total_weight

    return merged


def _pick_winning_branch(
    branches: list[dict],
    merged_scores: dict[str, float],
) -> dict[str, Any] | None:
    """Select the branch whose root_cause_candidate should be authoritative.

    Priority:
      1. Branches with confidence_level == "HIGH"
      2. Among those (or all if none are HIGH), pick the one whose
         root_cause_candidate component has the highest merged_score.
      3. If no branch has a root_cause_candidate, return the first branch.
    """
    candidates = [b for b in branches if b.get("root_cause_candidate") is not None]
    if not candidates:
        return branches[0] if branches else None

    high_confidence = [b for b in candidates if b.get("confidence_level") == "HIGH"]
    pool = high_confidence if high_confidence else candidates

    def branch_score(b: dict) -> float:
        rc = b.get("root_cause_candidate")
        if not rc:
            return 0.0
        component = _extract_component_id(rc)
        # Look for any hypothesis score that mentions this component
        for hyp, score in merged_scores.items():
            if component and component.lower() in hyp.lower():
                return score
        return 0.0

    return max(pool, key=branch_score)


def _extract_component_id(root_cause: dict | None) -> str | None:
    if not root_cause or not isinstance(root_cause, dict):
        return None
    return (
        root_cause.get("component_id")
        or root_cause.get("cmdb_id")
        or root_cause.get("component")
        or root_cause.get("name")
    )


def _deduplicate_pairs(
    items: list[dict],
    key_fields: tuple[str, str],
) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    f1, f2 = key_fields
    for item in items:
        key = (item.get(f1), item.get(f2))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _worst_case_investigation_state(branches: list[dict]) -> str:
    worst = "active"
    worst_rank = _INV_STATE_RANK.get("active", 3)
    for b in branches:
        state_val = b.get("investigation_state", "active")
        rank = _INV_STATE_RANK.get(state_val, 3)
        if rank < worst_rank:
            worst_rank = rank
            worst = state_val
    return worst


def _worst_case_confidence(branches: list[dict]) -> str | None:
    confidence_vals = [b.get("confidence_level") for b in branches if b.get("confidence_level")]
    if not confidence_vals:
        return None
    return min(confidence_vals, key=lambda c: _CONFIDENCE_RANK.get(c, 3))


def _build_final_report(
    winning_branch: dict | None,
    all_branches: list[dict],
    root_cause_candidate: dict | None,
) -> dict[str, Any] | None:
    base_report = (winning_branch or {}).get("final_report") or {}

    branch_summaries = []
    for b in all_branches:
        target = b.get("target_component", "ALL")
        rc = b.get("root_cause_candidate")
        conf = b.get("confidence_level", "UNKNOWN")
        branch_summaries.append({
            "component": target,
            "root_cause": _extract_component_id(rc),
            "confidence": conf,
            "evidence_count": len(b.get("evidence_items", [])),
            "narrative": b.get("narrative_summary", ""),
        })

    return {
        **base_report,
        "parallel_branch_count": len(all_branches),
        "root_cause_candidate": root_cause_candidate,
        "cross_branch_summaries": branch_summaries,
    }
