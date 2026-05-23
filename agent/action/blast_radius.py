"""
agent/action/blast_radius.py

Blast radius estimator for the AIRS Action Layer.

Before executing any automated remediation, the AIRS must calculate the exact
scope and potential collateral damage of the proposed change. This module
traverses the Enterprise Knowledge Graph to:

  1. Find all services that transitively depend on the target service.
  2. Identify which of those services are business-critical (tier 1).
  3. Map all propagation paths from the target to tier-1 assets.
  4. Compute a composite risk score (0.0–1.0).
  5. Output a recommendation: auto_execute | require_approval | block.

A blast radius that includes ANY tier-1 service automatically triggers
the human-in-the-loop approval gate, regardless of the overall risk score.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.reasoning.graph_schema import BlastRadiusReport
from agent.reasoning.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class BlastRadiusEstimator:
    """
    Blast radius estimator that traverses the EKG to calculate the full
    impact scope of a proposed remediation action.

    Usage:
        estimator = BlastRadiusEstimator()
        report = await estimator.estimate(
            target_service="payments-service",
            proposed_action="kubectl rollout restart deployment/payments-service",
        )
        if report.tier1_impact:
            # Route to human approval
    """

    def __init__(self) -> None:
        self._ekg = KnowledgeGraph.get_instance()

    async def estimate(
        self,
        target_service: str,
        proposed_action: str = "",
    ) -> BlastRadiusReport:
        """
        Calculate the blast radius of an action targeting ``target_service``.

        Args:
            target_service: The service being restarted, scaled, or patched.
            proposed_action: Human-readable description of the proposed action
                             (used only for logging).

        Returns:
            BlastRadiusReport with risk score and recommendation.
        """
        logger.info(
            "[BlastRadius] Estimating impact for target=%s action=%r",
            target_service, proposed_action[:80],
        )
        await self._ekg.initialize()
        report = await self._ekg.calculate_blast_radius(target_service)

        logger.info(
            "[BlastRadius] target=%s risk=%.2f tier1_impact=%s affected=%d recommendation=%s",
            target_service,
            report.risk_score,
            report.tier1_impact,
            len(report.affected_services),
            report.recommendation,
        )
        return report

    async def should_auto_execute(
        self,
        target_service: str,
    ) -> tuple[bool, BlastRadiusReport]:
        """
        Convenience method: returns (can_auto_execute, report).

        Auto-execution is permitted only when:
          - No tier-1 services are in the blast radius, AND
          - Risk score is below 0.3.

        Args:
            target_service: Service being modified.

        Returns:
            Tuple of (auto_execute_ok: bool, report: BlastRadiusReport).
        """
        report = await self.estimate(target_service)
        auto_ok = (not report.tier1_impact) and (report.risk_score < 0.3)
        return auto_ok, report
