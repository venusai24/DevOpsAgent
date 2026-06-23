"""
Diagnosis Activity — Module 1.10.

Temporal activity: Produce the final structured root cause analysis report.
Called when the route_decision emits DIAGNOSE.

The LLM sees:
  - The full causal chain from the Investigation Graph
  - All remaining active evidence
  - The leading hypothesis with its confidence
  - All considered hypotheses (confirmed, refuted, open)

Returns: InvestigationResult with root cause, confidence, and remediation recommendations.

Temporal contract:
  - Final terminal activity — no further state mutations needed after this
  - Returns InvestigationResult (not InvestigationState)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from temporalio import activity

from airs.config import settings
from airs.graph.prompts.interpretation import build_diagnosis_prompt
from airs.graph.prompts.tier1_constitution import get_constitution_prompt
from airs.models.investigation import InvestigationState, InvestigationStatus
from airs.models.results import (
    BlastRadius,
    InvestigationResult,
    RecommendedAction,
    ResultStatus,
)

log = logging.getLogger(__name__)


@activity.defn(name="produce_diagnosis")
async def produce_diagnosis(
    state: InvestigationState,
) -> InvestigationResult:
    """
    Call the LLM to produce the final root cause analysis.

    Args:
        state: Final InvestigationState before diagnosis.

    Returns:
        InvestigationResult with root_cause, confidence, causal_chain,
        blast_radius, and recommended_actions.
    """
    activity.logger.info(
        "Producing final diagnosis for investigation=%s at hop=%d",
        state.investigation_id, state.total_hop_count,
    )

    from airs.context.manager import ProactiveContextManager
    ctx_manager = ProactiveContextManager.from_settings()
    context_window = ctx_manager.get_context_window(state)

    # Build causal chain summary for prompt
    causal_nodes = context_window.get("causal_chain", [])
    causal_summary = "\n".join(
        f"  Hop {n.get('hop')}: [{n.get('entity')}] {n.get('signal')} — "
        f"score={n.get('context_score', 0):.2f}"
        for n in causal_nodes
    ) or "No confirmed causal chain yet."

    leading = state.get_leading_hypothesis()
    leading_text = (
        f"{leading.statement} (confidence={leading.confidence:.2f})"
        if leading else "No leading hypothesis."
    )

    hypotheses_summary = _format_all_hypotheses(state)

    diagnosis_prompt = build_diagnosis_prompt(
        alert_name=state.alert.alert_name,
        service=state.alert.service or "unknown",
        namespace=state.alert.namespace or "default",
        hop_index=state.total_hop_count,
        leading_hypothesis=leading_text,
        causal_chain_summary=causal_summary,
        hypotheses_summary=hypotheses_summary,
        trajectory_risk=state.risk_state.trajectory_risk_score,
        missing_mass=state.pursuit_state.current_missing_mass,
    )

    from litellm import completion
    response = completion(
        model=settings.primary_model,
        messages=[
            {"role": "system", "content": get_constitution_prompt()},
            {"role": "user", "content": diagnosis_prompt},
        ],
        temperature=0.05,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    result = _parse_diagnosis(raw, state)

    activity.logger.info(
        "Diagnosis complete: root_cause=%s, confidence=%.2f",
        result.root_cause[:80], result.confidence,
    )

    return result


def _parse_diagnosis(raw: str, state: InvestigationState) -> InvestigationResult:
    """Parse LLM JSON output into InvestigationResult."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Diagnosis LLM returned invalid JSON — using fallback")
        data = {}

    # Extract fields with fallbacks
    root_cause = data.get("root_cause", "Root cause undetermined — see evidence graph.")
    confidence = float(data.get("confidence", 0.5))

    # Parse blast radius
    br_data = data.get("blast_radius", {})
    blast_radius = BlastRadius(
        affected_services=br_data.get("affected_services", [state.alert.service or "unknown"]),
        estimated_user_impact=br_data.get("estimated_user_impact", "Unknown impact"),
    )

    # Parse recommended actions
    actions = []
    for i, a in enumerate(data.get("recommended_actions", []), 1):
        actions.append(RecommendedAction(
            priority=a.get("priority", i),
            action=a.get("action", "Investigate further"),
            tool=a.get("tool", ""),
            risk_level=a.get("risk_level", "medium"),
        ))

    # Extract causal chain from graph
    causal_chain = [
        {
            "node_id": nid,
            "hop": state.graph.nodes[nid].hop_index if nid in state.graph.nodes else -1,
        }
        for nid in state.graph.causal_chain_ids
    ]

    return InvestigationResult(
        investigation_id=state.investigation_id,
        alert_id=state.alert.alert_id,
        status=ResultStatus.RESOLVED,
        root_cause=root_cause,
        confidence=confidence,
        causal_chain=causal_chain,
        blast_radius=blast_radius,
        recommended_actions=actions,
        total_hops=state.total_hop_count,
        trajectory_risk=state.risk_state.trajectory_risk_score,
        final_missing_mass=state.pursuit_state.current_missing_mass,
        completed_at=datetime.now(timezone.utc),
    )


def _format_all_hypotheses(state: InvestigationState) -> str:
    if not state.hypotheses:
        return "No hypotheses."
    lines = []
    for h in state.hypotheses[:5]:
        lines.append(
            f"  [{h.status.value}] conf={h.confidence:.2f}: {h.statement}"
        )
    return "\n".join(lines)
