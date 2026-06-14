"""
airs_v2/action/confidence.py
==============================

Composite confidence engine for HITL routing (Stage 5).

Purpose
-------
The ConfidenceEngine computes a single composite score from three
independent evidence sources:

  1. Diagnosis confidence  (base quality of the reasoning engine's output)
  2. RAG confidence boost  (quality and quantity of retrieved precedents)
  3. Policy pass rate      (fraction of actions approved by the PolicyEnvelope)

The composite score determines whether an action can be executed autonomously
or must be routed to the Slack HITL gateway.

HITL Phase isolation invariant
-------------------------------
This module is called ONLY within the Action phase (Stage 4).
The ReasoningEngine (Stage 3) does NOT call this — it computes the base
confidence independently. HITL is NEVER triggered by this engine itself;
it only returns a boolean signal that the Executor uses to decide routing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default confidence threshold — overridden by config.py ACTION_CONFIDENCE_THRESHOLD
_DEFAULT_THRESHOLD = 0.75


class ConfidenceEngine:
    """
    Composite confidence engine for HITL routing.

    Weights
    -------
    W_DIAG   = 0.60  — Diagnosis confidence (primary signal)
    W_RAG    = 0.25  — RAG boost normalised to [0,1] (evidence quality)
    W_POLICY = 0.15  — Policy pass rate (action safety signal)

    Parameters
    ----------
    threshold : Minimum composite score for autonomous execution.
                Below this value, HITL is required.
    """

    W_DIAG: float = 0.60
    W_RAG: float = 0.25
    W_POLICY: float = 0.15

    # Max RAG boost from RAGEngine.compute_rag_confidence_boost()
    # Used to normalise [0, 0.25] → [0, 1] for the weighted formula
    _RAG_BOOST_MAX: float = 0.25

    def __init__(self, threshold: float | None = None) -> None:
        from config import settings

        self._threshold = threshold or settings.ACTION_CONFIDENCE_THRESHOLD
        logger.debug(
            "[ConfidenceEngine] threshold=%.2f", self._threshold
        )

    def compute_composite_confidence(
        self,
        diagnosis_confidence: float,
        rag_boost: float,
        policy_pass_rate: float,
    ) -> float:
        """
        Compute the composite confidence score.

        Parameters
        ----------
        diagnosis_confidence : IncidentAnalysis.overall_confidence (0.0–1.0)
        rag_boost            : RAGEngine.compute_rag_confidence_boost() (0.0–0.25)
        policy_pass_rate     : Fraction of actions AUTO_APPROVED by PolicyEnvelope (0.0–1.0)

        Returns
        -------
        Composite confidence in [0.0, 1.0].
        """
        # Normalise RAG boost from [0, 0.25] → [0, 1]
        rag_normalised = min(1.0, rag_boost / self._RAG_BOOST_MAX) if rag_boost > 0 else 0.0

        composite = (
            diagnosis_confidence * self.W_DIAG
            + rag_normalised * self.W_RAG
            + policy_pass_rate * self.W_POLICY
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        logger.debug(
            "[ConfidenceEngine] diag=%.3f rag_norm=%.3f policy=%.3f → composite=%.4f",
            diagnosis_confidence,
            rag_normalised,
            policy_pass_rate,
            composite,
        )
        return composite

    def requires_hitl(self, composite: float) -> bool:
        """
        Return True if the composite score is below the HITL threshold.

        True  → route to Slack gateway for human approval.
        False → autonomous execution is permitted.
        """
        return composite < self._threshold

    def compute_policy_pass_rate(self, policy_decisions: list) -> float:
        """
        Compute the fraction of PolicyDecision objects that are AUTO_APPROVED.

        Parameters
        ----------
        policy_decisions : list of PolicyDecision objects.
        """
        from airs_v2.action.types import DecisionOutcome

        if not policy_decisions:
            return 0.0
        approved = sum(
            1 for d in policy_decisions
            if d.decision == DecisionOutcome.AUTO_APPROVED
        )
        return round(approved / len(policy_decisions), 4)
