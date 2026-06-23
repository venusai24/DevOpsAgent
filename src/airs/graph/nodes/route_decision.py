"""
Route Decision Node — Module 1.9.

The LangGraph conditional edge function that routes the investigation
to the correct next activity based on the current InvestigationState.

Route logic (deterministic — no LLM):
  1. Max hops exceeded               → ESCALATE
  2. Supermartingale M_t >= 1/delta  → ESCALATE
  3. Missing mass < epsilon          → check hypothesis confidence
     3a. Leading hypothesis conf > 0.85 → DIAGNOSE
     3b. Otherwise                      → QUERY_PLAYBOOK
  4. Consecutive low delta >= 3      → QUERY_PLAYBOOK
  5. Default                         → EXECUTE_TOOL (continue investigation)

This is a pure routing function — it reads state but never modifies it.
The actual LLM call happens in the reasoning_activity.
"""
from __future__ import annotations

import logging
from typing import Literal

from airs.models.hypothesis import HypothesisStatus
from airs.models.investigation import InvestigationState

log = logging.getLogger(__name__)

RouteDecision = Literal[
    "execute_tool",
    "query_playbook",
    "diagnose",
    "escalate",
]

# Hypothesis confidence threshold for DIAGNOSE transition
_DIAGNOSE_CONFIDENCE_THRESHOLD = 0.85


def route_decision(state: InvestigationState, max_hops: int = 50) -> RouteDecision:
    """
    Deterministic routing function for the LangGraph conditional edge.

    Args:
        state:    Current InvestigationState.
        max_hops: Maximum allowed hops before forced escalation.

    Returns:
        RouteDecision string — the name of the next LangGraph node to execute.
    """
    risk = state.risk_state
    pursuit = state.pursuit_state

    # ── Rule 1: Max hops exceeded ─────────────────────────────────────────────
    if state.total_hop_count >= max_hops:
        log.warning(
            "Max hops (%d) reached at hop %d — escalating",
            max_hops, state.total_hop_count,
        )
        return "escalate"

    # ── Rule 2: Supermartingale alarm ─────────────────────────────────────────
    if risk.should_escalate:
        log.warning(
            "Supermartingale M_t=%.3f >= 1/δ=%.3f — escalating",
            risk.supermartingale_value, risk.escalation_threshold,
        )
        return "escalate"

    # ── Rule 3: Missing mass converged ────────────────────────────────────────
    if pursuit.is_complete:
        # Check if we have a high-confidence hypothesis
        confirmed = _get_confirmed_hypothesis(state)
        if confirmed and confirmed.confidence >= _DIAGNOSE_CONFIDENCE_THRESHOLD:
            log.info(
                "Missing mass converged (%.3f) + hypothesis confidence %.2f — diagnosing",
                pursuit.current_missing_mass, confirmed.confidence,
            )
            return "diagnose"
        else:
            log.info(
                "Missing mass converged but hypothesis confidence insufficient — querying playbook",
            )
            return "query_playbook"

    # ── Rule 4: Plateau detection ─────────────────────────────────────────────
    if pursuit.is_plateau:
        log.info(
            "Information pursuit plateau detected (%d consecutive low-delta hops) — querying playbook",
            pursuit.consecutive_low_delta,
        )
        return "query_playbook"

    # ── Rule 5: High-confidence diagnosis shortcut ────────────────────────────
    leading = state.get_leading_hypothesis()
    if leading and leading.confidence >= 0.92:
        log.info(
            "High-confidence hypothesis (%.2f) — early diagnosis",
            leading.confidence,
        )
        return "diagnose"

    # ── Default: continue investigation ───────────────────────────────────────
    return "execute_tool"


def _get_confirmed_hypothesis(state: InvestigationState):
    """Return the confirmed/leading hypothesis if one exists."""
    from airs.models.hypothesis import HypothesisStatus
    leading = state.get_leading_hypothesis()
    if leading and leading.status in (HypothesisStatus.LEADING, HypothesisStatus.CONFIRMED):
        return leading
    # Fall back to highest-confidence active hypothesis
    active = state.get_active_hypotheses()
    if active:
        return max(active, key=lambda h: h.confidence)
    return None
