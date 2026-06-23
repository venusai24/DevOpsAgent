"""
Initialize Investigation Activity — Module 1.10.

Temporal activity: Bootstrap a new investigation from an incoming alert.
Creates the initial InvestigationState, seeds the first hypothesis,
and sets up the entity resolution context.

Temporal contract:
  - Idempotent: safe to retry (same alert_id always produces same initial state)
  - No external writes: only builds in-memory state
  - Returns: InvestigationState (serialized by Temporal to workflow history)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from temporalio import activity

from airs.models.context import ContextBudget, InsightTiers
from airs.models.hypothesis import Hypothesis, HypothesisStatus
from airs.models.investigation import (
    Alert,
    InvestigationGraph,
    InvestigationState,
    InvestigationStatus,
)
from airs.models.pursuit import InformationPursuitState
from airs.models.risk import TrajectoryRiskState

log = logging.getLogger(__name__)


@activity.defn(name="initialize_investigation")
async def initialize_investigation(
    alert_id: str,
    alert_name: str,
    description: str,
    service: str,
    namespace: str,
    severity: str,
    raw_alert_payload: dict,
) -> InvestigationState:
    """
    Bootstrap a new investigation from an incoming alert.

    Args:
        alert_id:           Unique alert identifier (from Datadog/Prometheus).
        alert_name:         Human-readable alert name.
        description:        Alert description / trigger condition.
        service:            Primary affected service name.
        namespace:          Kubernetes namespace.
        severity:           Severity level ('critical', 'high', 'medium', 'low').
        raw_alert_payload:  Complete raw alert dict for provenance.

    Returns:
        Initial InvestigationState ready for the first reasoning hop.
    """
    activity.logger.info(
        "Initializing investigation for alert=%s service=%s severity=%s",
        alert_name, service, severity,
    )

    alert = Alert(
        alert_id=alert_id,
        alert_name=alert_name,
        description=description,
        service=service,
        namespace=namespace,
        severity=severity,
        raw_payload=raw_alert_payload,
        received_at=datetime.now(timezone.utc),
    )

    # Seed initial hypothesis from alert context
    initial_hypothesis = Hypothesis(
        hypothesis_id=str(uuid.uuid4()),
        statement=(
            f"Service '{service}' is experiencing a '{alert_name}' condition "
            f"in namespace '{namespace}'."
        ),
        confidence=0.10,  # Very low — no evidence yet
        status=HypothesisStatus.ACTIVE,
        supporting_node_ids=[],
        contradicting_node_ids=[],
        created_at_hop=0,
        last_updated_hop=0,
    )

    # Load calibration settings
    from airs.config import settings
    from airs.risk.calibration import get_calibration_store
    calibration = get_calibration_store()

    # Build initial risk state
    delta = calibration.get_escalation_delta()
    risk_state = TrajectoryRiskState(
        trajectory_risk_score=0.0,
        step_risks=[],
        supermartingale_value=1.0,  # M_0 = 1 always
        escalation_threshold=1.0 / delta,
        lambda_threshold=calibration.get_lambda_threshold(),
        should_escalate=False,
    )

    # Build initial pursuit state
    pursuit_state = InformationPursuitState(
        current_missing_mass=1.0,  # Start with maximum uncertainty
        previous_missing_mass=1.0,
        missing_mass_delta=0.0,
        epsilon_threshold=settings.epsilon_threshold,
        consecutive_low_delta=0,
        filled_evidence_categories=[],
        open_evidence_categories=[],  # Will be populated on first missing_mass calculation
    )

    state = InvestigationState(
        investigation_id=alert_id,  # Reuse alert_id as investigation_id
        alert=alert,
        status=InvestigationStatus.IN_PROGRESS,
        hypotheses=[initial_hypothesis],
        leading_hypothesis_id=initial_hypothesis.hypothesis_id,
        graph=InvestigationGraph(
            nodes={},
            edges=[],
            causal_chain_ids=[],
        ),
        insight_tiers=InsightTiers(
            active=[],
            summarized=[],
            archived=[],
            discarded_count=0,
        ),
        context_budget=ContextBudget(),
        risk_state=risk_state,
        pursuit_state=pursuit_state,
        total_hop_count=0,
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )

    activity.logger.info(
        "Investigation initialized: id=%s, initial_hypothesis=%s",
        state.investigation_id,
        initial_hypothesis.hypothesis_id,
    )

    return state
