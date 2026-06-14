"""
airs_v2/action/types.py
=======================

Pydantic v2 data models for the Stage 4 Execution Engine.

Type hierarchy
--------------
ActionKind            — Enum of allowed remediation action kinds.
RiskLevel             — Enum of risk levels (LOW → CRITICAL).
DecisionOutcome       — Enum: AUTO_APPROVED | PENDING_HUMAN | REJECTED.
RemediationAction     — A single atomic remediation step (never directly executed).
RemediationPlan       — Ordered list of RemediationAction items for one incident.
PolicyViolation       — Machine-readable rule violation record.
PolicyDecision        — Per-action policy evaluation result.
HealthCheckResult     — Post-execution probe outcome for one service.
ExecutionResult       — Full audit trail for one plan execution run.

Safety invariants
-----------------
* ``RemediationAction`` is a *description* of an action, not an invocable.
  The engine emits these as audit records; external actuators interpret them.
* ``risk_level == CRITICAL`` is always REJECTED by the PolicyEnvelope — there
  is no code path that leads to AUTO_APPROVED for CRITICAL actions.
* ``kind == exec_script`` is always treated as HIGH risk by PE-R5, even if
  the caller declared LOW or MEDIUM.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionKind(str, enum.Enum):
    """The set of recognised, policy-auditable remediation action kinds."""

    RESTART_POD = "restart_pod"
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    DRAIN_NODE = "drain_node"
    EXEC_SCRIPT = "exec_script"


class RiskLevel(str, enum.Enum):
    """
    Risk level declared by the runbook author.

    The PolicyEnvelope may upgrade a risk level (e.g. exec_script LOW → HIGH)
    but never downgrades one.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DecisionOutcome(str, enum.Enum):
    """
    The three possible outcomes from the PolicyEnvelope for a single action.

    AUTO_APPROVED   — All rules passed; action may proceed without human review.
    PENDING_HUMAN   — One or more rules require human sign-off (routed to Slack).
    REJECTED        — A hard rule violation; the action must not proceed.
    """

    AUTO_APPROVED = "AUTO_APPROVED"
    PENDING_HUMAN = "PENDING_HUMAN"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Core action model
# ---------------------------------------------------------------------------


class RemediationAction(BaseModel):
    """
    A single atomic remediation step described declaratively.

    This model is a *description*, not an invocable.  The Execution Engine
    routes it through the policy layer and records the decision; it never
    shells out, calls kubectl, or invokes boto3.

    Attributes
    ----------
    action_id:
        Stable UUID for this action instance.  Auto-generated if not supplied.
    kind:
        The type of remediation to perform (``ActionKind``).
    target_namespace:
        Kubernetes namespace (or equivalent logical namespace) in which the
        action would operate.
    target_resource:
        Name of the resource to act on (pod name, deployment name, node name,
        or script path for exec_script).
    parameters:
        Arbitrary key-value metadata for the action.  The ``"emergency"`` key
        (bool) disables PE-R3 business-hours enforcement for this action.
    risk_level:
        Declared risk level from the runbook author.  PE-R5 may upgrade
        exec_script actions from any level to HIGH.
    estimated_blast_radius:
        Estimated number of pods that would be affected.  Used by PE-R2.
    runbook_ref:
        Human-readable reference to the original runbook or SOP (for audit).
    """

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable UUID for this action instance.",
    )
    kind: ActionKind
    target_namespace: str
    target_resource: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_blast_radius: int = Field(
        default=1,
        ge=0,
        description="Estimated number of pods affected (0 = zero impact).",
    )
    runbook_ref: str = ""


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------


class RemediationPlan(BaseModel):
    """
    An ordered list of RemediationAction items addressing a single incident.

    Each action is evaluated independently by the PolicyEnvelope; approval of
    one action does not imply approval of the others.

    Attributes
    ----------
    plan_id:
        Stable UUID for this plan.
    incident_id:
        Reference to the originating incident (from IncidentAnalysis or external
        ticketing system).
    actions:
        Ordered list of RemediationAction objects.  The engine evaluates each
        independently.
    created_at:
        ISO-8601 UTC timestamp when the plan was created.
    """

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str = ""
    actions: list[RemediationAction] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Policy decision models
# ---------------------------------------------------------------------------


class PolicyViolation(BaseModel):
    """
    A single, machine-readable rule violation for one action.

    Attributes
    ----------
    rule_id:
        Short machine-readable rule identifier, e.g. ``"PE-R1"``.
    rule_name:
        Human-readable rule name, e.g. ``"NAMESPACE_ALLOWLIST"``.
    action_id:
        The ``action_id`` of the ``RemediationAction`` that violated this rule.
    reason:
        Descriptive explanation of why the rule was violated.
    """

    rule_id: str
    rule_name: str
    action_id: str
    reason: str


class PolicyDecision(BaseModel):
    """
    The result of evaluating one ``RemediationAction`` through the PolicyEnvelope.

    Attributes
    ----------
    action_id:
        The evaluated action's ``action_id``.
    action_kind:
        The evaluated action's ``kind`` (copied for convenience).
    effective_risk_level:
        The risk level *after* PE-R5 may have upgraded it (e.g. exec_script LOW→HIGH).
    approved:
        True iff ``decision == AUTO_APPROVED``.
    decision:
        The three-way outcome: AUTO_APPROVED, PENDING_HUMAN, or REJECTED.
    violations:
        All rule violations accumulated for this action.  May be non-empty even
        when ``decision == PENDING_HUMAN`` (PE-R4 routes rather than rejects).
    slack_message_ts:
        The mock Slack message timestamp, set when decision is PENDING_HUMAN.
        Empty string if not routed.
    """

    action_id: str
    action_kind: str
    effective_risk_level: RiskLevel
    approved: bool
    decision: DecisionOutcome
    violations: list[PolicyViolation] = Field(default_factory=list)
    slack_message_ts: str = ""


# ---------------------------------------------------------------------------
# Health check model
# ---------------------------------------------------------------------------


class HealthCheckResult(BaseModel):
    """
    Outcome of one post-execution health probe for one service.

    Attributes
    ----------
    service:
        Service name probed.
    healthy:
        True if the service reported healthy.
    http_status:
        HTTP status code from the readiness/liveness probe (0 if unreachable).
    latency_ms:
        Round-trip latency in milliseconds (0.0 if unreachable).
    error_message:
        Non-empty when healthy=False, describing the failure.
    checked_at:
        ISO-8601 UTC timestamp of the probe.
    """

    service: str
    healthy: bool
    http_status: int = 200
    latency_ms: float = 0.0
    error_message: str = ""
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Execution result model
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    """
    Full audit record produced by ``ExecutionEngine.execute_plan()``.

    This is the single source of truth for what happened during a plan
    execution attempt.  It contains no code outputs because the engine
    never executes code directly.

    Attributes
    ----------
    plan_id:
        The ``plan_id`` of the evaluated ``RemediationPlan``.
    incident_id:
        Carried over from the plan.
    policy_decisions:
        One ``PolicyDecision`` per action, in plan order.
    actions_approved:
        action_ids of actions that received AUTO_APPROVED.
    actions_pending_human:
        action_ids of actions routed to Slack for human approval.
    actions_rejected:
        action_ids of actions hard-rejected by the policy engine.
    health_checks:
        List of post-execution health probe results.
    rollback_triggered:
        True if any health check returned healthy=False.
    rollback_reason:
        Machine-readable reason string, set when rollback_triggered=True.
    completed_at:
        ISO-8601 UTC timestamp when the execution result was produced.
    """

    plan_id: str
    incident_id: str = ""
    policy_decisions: list[PolicyDecision] = Field(default_factory=list)
    actions_approved: list[str] = Field(default_factory=list)
    actions_pending_human: list[str] = Field(default_factory=list)
    actions_rejected: list[str] = Field(default_factory=list)
    health_checks: list[HealthCheckResult] = Field(default_factory=list)
    rollback_triggered: bool = False
    rollback_reason: str = ""
    completed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Stage 5: HITL / RAG audit fields ─────────────────────────────────────
    # Populated by execute_plan_with_confidence(). Defaults preserve backwards
    # compatibility when using the legacy execute_plan() path.
    composite_confidence: float = 0.0
    """Composite confidence score from ConfidenceEngine (0.0–1.0)."""
    rag_context_used: bool = False
    """True if a RAGEngine was active and returned at least one result."""
    hitl_triggered: bool = False
    """True if composite_confidence < threshold and Slack HITL was triggered."""
    human_feedback: Optional[Any] = None
    """HumanFeedback received from the Slack/CLI gateway (None for auto-approved)."""
