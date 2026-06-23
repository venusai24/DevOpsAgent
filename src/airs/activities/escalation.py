"""
Escalation Activity — Module 1.10.

Temporal activity: Produce a structured escalation handoff report when
the supermartingale alarm fires or the hop budget is exhausted.

Returns InvestigationResult(status=ESCALATED) with:
  - Best available hypothesis
  - All evidence gathered so far
  - Suggested next steps for the on-call engineer
  - Urgency level
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from temporalio import activity

from airs.config import settings
from airs.graph.prompts.interpretation import build_escalation_prompt
from airs.graph.prompts.tier1_constitution import get_constitution_prompt
from airs.models.investigation import InvestigationState
from airs.models.results import (
    BlastRadius,
    InvestigationResult,
    RecommendedAction,
    ResultStatus,
)

log = logging.getLogger(__name__)


@activity.defn(name="produce_escalation")
async def produce_escalation(
    state: InvestigationState,
    escalation_reason: str,
) -> InvestigationResult:
    """
    Produce an escalation handoff report.

    Args:
        state:              Final InvestigationState before escalation.
        escalation_reason:  Human-readable reason (e.g., 'Supermartingale alarm').

    Returns:
        InvestigationResult(status=ESCALATED) with best-available hypothesis
        and suggested next steps.
    """
    activity.logger.info(
        "Producing escalation report for investigation=%s (reason=%s)",
        state.investigation_id, escalation_reason,
    )

    risk = state.risk_state
    leading = state.get_leading_hypothesis()
    leading_text = (
        f"{leading.statement} (confidence={leading.confidence:.2f})"
        if leading else "No hypothesis established."
    )

    # Summarize gathered evidence
    active_nodes = state.insight_tiers.active
    evidence_summary = "\n".join(
        f"  Hop {n.hop_index} | {n.entity.entity_type.value} | {n.signal_source.value} | "
        f"score={n.context_score or 0:.2f}"
        for n in active_nodes[:10]
    ) or "No evidence gathered."

    escalation_prompt = build_escalation_prompt(
        alert_name=state.alert.alert_name,
        service=state.alert.service or "unknown",
        hop_index=state.total_hop_count,
        escalation_reason=escalation_reason,
        supermartingale=risk.supermartingale_value,
        trajectory_risk=risk.trajectory_risk_score,
        leading_hypothesis=leading_text,
        evidence_summary=evidence_summary,
    )

    from litellm import completion
    response = completion(
        model=settings.primary_model,
        messages=[
            {"role": "system", "content": get_constitution_prompt()},
            {"role": "user", "content": escalation_prompt},
        ],
        temperature=0.1,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    result = _parse_escalation(raw, state, escalation_reason)

    activity.logger.info(
        "Escalation report produced: urgency=%s, steps=%d",
        result.escalation_urgency, len(result.recommended_actions),
    )

    return result


def _parse_escalation(
    raw: str, state: InvestigationState, reason: str
) -> InvestigationResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    summary = data.get("escalation_summary", f"Investigation escalated: {reason}")
    urgency = data.get("urgency", "high")

    actions = [
        RecommendedAction(priority=i + 1, action=step, tool="", risk_level="medium")
        for i, step in enumerate(data.get("suggested_next_steps", []))
    ]

    leading = state.get_leading_hypothesis()
    blast_radius = BlastRadius(
        affected_services=[state.alert.service or "unknown"],
        estimated_user_impact="Requires human investigation — see escalation report.",
    )

    return InvestigationResult(
        investigation_id=state.investigation_id,
        alert_id=state.alert.alert_id,
        status=ResultStatus.ESCALATED,
        root_cause=summary,
        confidence=leading.confidence if leading else 0.0,
        causal_chain=[],
        blast_radius=blast_radius,
        recommended_actions=actions,
        total_hops=state.total_hop_count,
        trajectory_risk=state.risk_state.trajectory_risk_score,
        final_missing_mass=state.pursuit_state.current_missing_mass,
        escalation_urgency=urgency,
        escalation_reason=reason,
        completed_at=datetime.now(timezone.utc),
    )
