"""
airs_v2/action/executor.py
===========================

ExecutionEngine — Policy-Gated Remediation Orchestrator
---------------------------------------------------------

The ExecutionEngine is the top-level entry point for Stage 4.  It:

1. Receives a ``RemediationPlan`` (an ordered list of declarative actions).
2. Runs each action through the ``PolicyEnvelope`` (5-rule deterministic gate).
3. Routes ``PENDING_HUMAN`` decisions to the ``MockSlackGateway``.
4. Collects ``AUTO_APPROVED`` decisions into an audit-only approval list.
5. Immediately hard-stops ``REJECTED`` actions (they are never dispatched).
6. Runs post-execution health checks via ``HealthMonitor``.
7. If **any** health check fails, sets ``rollback_triggered=True`` in the
   ``ExecutionResult`` and records the reason.

Critical safety guarantee
--------------------------
The engine **never executes code directly**.  It:
- Does NOT shell out.
- Does NOT call kubectl.
- Does NOT call boto3, the k8s Python SDK, or any cloud API.
- Does NOT invoke the action's ``parameters`` as a callable.

``ExecutionResult.actions_approved`` is a list of **action IDs** (strings),
not a list of execution outputs.  External actuators (human or CI pipeline)
read the result and decide what to do with it.

Usage
-----
::

    from datetime import datetime, timezone
    from airs_v2.action.executor import ExecutionEngine
    from airs_v2.action.types import RemediationPlan, RemediationAction, ActionKind, RiskLevel

    engine = ExecutionEngine()
    result = await engine.execute_plan(
        plan,
        now=datetime(2026, 6, 6, 10, 0, tzinfo=timezone.utc),   # business hours
        health_overrides={"payments-service": True},              # all healthy
    )
    assert result.rollback_triggered is False
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from airs_v2.action.health_monitor import HealthMonitor
from airs_v2.action.policy_envelope import PolicyEnvelope
from airs_v2.action.slack_gateway import MockSlackGateway
from airs_v2.action.types import (
    DecisionOutcome,
    ExecutionResult,
    PolicyDecision,
    RemediationPlan,
)
from airs_v2.evaluation.tracer import ObservabilityTracer

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Policy-gated remediation orchestrator.

    The engine coordinates the PolicyEnvelope, SlackGateway, and HealthMonitor
    into a single, auditable pipeline.

    Parameters
    ----------
    policy_envelope:
        The ``PolicyEnvelope`` instance to use.  Defaults to a fresh instance
        with the standard configuration (namespace allowlist, blast radius caps,
        business hours).  Inject a customised instance in tests.
    slack_gateway:
        The ``MockSlackGateway`` instance for routing high-risk actions.
        Inject a shared instance to inspect sent messages in tests.
    health_monitor:
        The ``HealthMonitor`` for post-execution probes.  Inject a shared
        instance to control probe outcomes in tests.
    affected_services:
        List of service names to health-check after plan evaluation.  If
        empty, health checks are derived from the unique target_resource values
        of all approved actions.
    """

    def __init__(
        self,
        *,
        policy_envelope: PolicyEnvelope | None = None,
        slack_gateway: MockSlackGateway | None = None,
        health_monitor: HealthMonitor | None = None,
        affected_services: list[str] | None = None,
    ) -> None:
        self._policy = policy_envelope or PolicyEnvelope()
        self._slack = slack_gateway or MockSlackGateway()
        self._health = health_monitor or HealthMonitor()
        self._affected_services = affected_services or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def slack_gateway(self) -> MockSlackGateway:
        """Expose the slack gateway for test assertions."""
        return self._slack

    async def execute_plan(
        self,
        plan: RemediationPlan,
        *,
        now: datetime | None = None,
        health_overrides: dict[str, bool] | None = None,
    ) -> ExecutionResult:
        """
        Evaluate a remediation plan through the full policy pipeline.

        Parameters
        ----------
        plan:
            The remediation plan to evaluate.
        now:
            Reference timestamp for business-hours evaluation.  Defaults to
            ``datetime.now(timezone.utc)``.  Inject in tests for determinism.
        health_overrides:
            Injected health-check results for testing.  Maps ``service_name →
            healthy (bool)``.  Services not listed default to healthy.

        Returns
        -------
        ExecutionResult
            Full audit trail including policy decisions, approval routing,
            and rollback trigger status.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        logger.info(
            "[ExecutionEngine] BEGIN plan_id=%s incident_id=%s actions=%d",
            plan.plan_id,
            plan.incident_id,
            len(plan.actions),
        )

        # ── ① Policy evaluation ────────────────────────────────────────────
        decisions: list[PolicyDecision] = self._policy.evaluate(plan, now=now)

        # ── ② Route decisions ──────────────────────────────────────────────
        approved: list[str] = []
        pending_human: list[str] = []
        rejected: list[str] = []

        # Build a quick lookup: action_id → action
        action_by_id = {a.action_id: a for a in plan.actions}

        for decision in decisions:
            action = action_by_id[decision.action_id]
            tracer = ObservabilityTracer.get_instance()

            if decision.decision == DecisionOutcome.AUTO_APPROVED:
                approved.append(decision.action_id)
                logger.info(
                    "[ExecutionEngine] AUTO_APPROVED action_id=%s kind=%s ns=%s",
                    decision.action_id,
                    decision.action_kind,
                    action.target_namespace,
                )
                tracer.record_action_taken(f"execute_{action.action_kind}", f"AUTO_APPROVED: {action.target_resource}")

            elif decision.decision == DecisionOutcome.PENDING_HUMAN:
                # ── ③ Route to Slack ─────────────────────────────────────
                message_ts = await self._slack.send_approval_request(
                    action, decision.violations
                )
                decision.slack_message_ts = message_ts
                pending_human.append(decision.action_id)
                logger.info(
                    "[ExecutionEngine] PENDING_HUMAN action_id=%s kind=%s "
                    "slack_ts=%s",
                    decision.action_id,
                    decision.action_kind,
                    message_ts,
                )
                tracer.record_action_taken(f"request_{action.action_kind}", f"PENDING_HUMAN: slack_ts={message_ts}")

            else:  # REJECTED
                rejected.append(decision.action_id)
                logger.warning(
                    "[ExecutionEngine] REJECTED action_id=%s kind=%s "
                    "violations=%s",
                    decision.action_id,
                    decision.action_kind,
                    [v.rule_id for v in decision.violations],
                )
                tracer.record_action_taken(f"rejected_{action.action_kind}", f"REJECTED: violations={[v.rule_id for v in decision.violations]}")

        # ── ④ Determine services to health-check ──────────────────────────
        probe_targets = self._resolve_probe_targets(plan, approved)

        # ── ⑤ Post-execution health checks ────────────────────────────────
        health_results = await self._health.run_checks(
            probe_targets,
            _inject_results=health_overrides,
        )

        # ── ⑥ Mandatory rollback trigger ──────────────────────────────────
        rollback_triggered = False
        rollback_reason = ""

        if HealthMonitor.any_unhealthy(health_results):
            rollback_triggered = True
            unhealthy = HealthMonitor.unhealthy_services(health_results)
            rollback_reason = (
                f"post_execution_health_check_failed: "
                f"services [{', '.join(unhealthy)}] reported unhealthy "
                f"after plan execution."
            )
            logger.error(
                "[ExecutionEngine] ROLLBACK TRIGGERED plan_id=%s "
                "unhealthy_services=%s",
                plan.plan_id,
                unhealthy,
            )

        # ── ⑦ Build audit result ───────────────────────────────────────────
        result = ExecutionResult(
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            policy_decisions=decisions,
            actions_approved=approved,
            actions_pending_human=pending_human,
            actions_rejected=rejected,
            health_checks=health_results,
            rollback_triggered=rollback_triggered,
            rollback_reason=rollback_reason,
        )

        logger.info(
            "[ExecutionEngine] DONE plan_id=%s approved=%d pending=%d "
            "rejected=%d rollback=%s",
            plan.plan_id,
            len(approved),
            len(pending_human),
            len(rejected),
            rollback_triggered,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_probe_targets(
        self, plan: RemediationPlan, approved_ids: list[str]
    ) -> list[str]:
        """
        Determine which services to health-check after plan evaluation.

        Priority:
        1. If ``self._affected_services`` was set at construction time, use it.
        2. Otherwise, collect unique ``target_resource`` values from approved
           actions as a best-effort proxy for affected services.
        3. If no approved actions exist, probe all target_resources in the plan
           (conservative: even rejected actions may have affected things).
        """
        if self._affected_services:
            return self._affected_services

        approved_set = set(approved_ids)
        targets: list[str] = []
        seen: set[str] = set()

        for action in plan.actions:
            resource = action.target_resource
            if resource and resource not in seen:
                # Only probe resources from approved actions (or all if none approved)
                if action.action_id in approved_set or not approved_set:
                    targets.append(resource)
                    seen.add(resource)

        return targets
