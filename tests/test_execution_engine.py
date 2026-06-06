"""
tests/test_execution_engine.py
================================

End-to-End Test Suite for Stage 4 — Execution Engine & Policy Envelope.

Test structure
--------------
TestPolicyEnvelopeRules
    Unit tests for each of the five policy rules in isolation.
    All five rules are proven to fire deterministically with the correct
    DecisionOutcome and rule ID.

TestSafeRunbook
    End-to-end tests proving that a compliant LOW-risk ``restart_pod``
    runbook flows through the full pipeline:
    - Receives AUTO_APPROVED
    - Does NOT route to Slack
    - Does NOT trigger rollback when all health checks pass
    - DOES trigger rollback when a health check fails

TestUnsafeRunbook
    End-to-end tests proving that non-compliant or high-risk actions are
    stopped by the policy engine:
    - exec_script always routed to Slack (PE-R5 + PE-R4)
    - Forbidden namespaces always REJECTED (PE-R1)
    - drain_node outside business hours REJECTED (PE-R3)
    - CRITICAL risk always REJECTED (PE-R4)
    - Mixed plan: per-action gating (not plan-level)

TestRollbackTrigger
    Proves the mandatory rollback invariant:
    - Fires if and only if any health check reports unhealthy
    - Does not fire when all healthy
    - Produces a descriptive rollback_reason

TestIronInvariants
    Cross-cutting invariant tests that must hold for ALL scenarios:
    1. CRITICAL risk → always REJECTED, no exceptions
    2. exec_script → always PENDING_HUMAN (never AUTO_APPROVED)
    3. Rollback fires iff any_unhealthy
    4. Slack called iff and only iff decision is PENDING_HUMAN
    5. actions_approved contains only action IDs (strings), never code outputs

All tests are fully deterministic: time is always injected, health checks
always use _inject_results, and no real network calls are made.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from airs_v2.action import (
    ActionKind,
    DecisionOutcome,
    ExecutionEngine,
    ExecutionResult,
    HealthMonitor,
    MockSlackGateway,
    PolicyEnvelope,
    PolicyDecision,
    RemediationAction,
    RemediationPlan,
    RiskLevel,
    PE_R1, PE_R2, PE_R3, PE_R4, PE_R5,
    PE_R1_NAME, PE_R2_NAME, PE_R3_NAME, PE_R4_NAME, PE_R5_NAME,
    DEFAULT_ALLOWED_NAMESPACES,
)


# ---------------------------------------------------------------------------
# Deterministic timestamps
# ---------------------------------------------------------------------------

# Monday 2026-06-08 10:30 UTC — inside business hours
BUSINESS_HOURS_TS = datetime(2026, 6, 8, 10, 30, tzinfo=timezone.utc)

# Saturday 2026-06-06 22:00 UTC — outside business hours
OUTSIDE_HOURS_TS = datetime(2026, 6, 6, 22, 0, tzinfo=timezone.utc)

# Monday 2026-06-08 08:00 UTC — before business hours (Mon but 08:00 < 09:00)
BEFORE_HOURS_TS = datetime(2026, 6, 8, 8, 0, tzinfo=timezone.utc)

# Friday 2026-06-12 17:00 UTC — exactly at close (exclusive end)
AT_CLOSE_TS = datetime(2026, 6, 12, 17, 0, tzinfo=timezone.utc)

# Friday 2026-06-12 16:59 UTC — one minute before close (inside)
NEAR_CLOSE_TS = datetime(2026, 6, 12, 16, 59, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _action(
    *,
    kind: ActionKind = ActionKind.RESTART_POD,
    namespace: str = "production",
    resource: str = "payments-service",
    risk: RiskLevel = RiskLevel.LOW,
    blast_radius: int = 2,
    params: dict | None = None,
    runbook_ref: str = "RB-001",
) -> RemediationAction:
    return RemediationAction(
        kind=kind,
        target_namespace=namespace,
        target_resource=resource,
        risk_level=risk,
        estimated_blast_radius=blast_radius,
        parameters=params or {},
        runbook_ref=runbook_ref,
    )


def _plan(*actions: RemediationAction, incident_id: str = "INC-TEST-001") -> RemediationPlan:
    return RemediationPlan(incident_id=incident_id, actions=list(actions))


def _engine(
    *,
    slack: MockSlackGateway | None = None,
    affected_services: list[str] | None = None,
) -> ExecutionEngine:
    slack = slack or MockSlackGateway()
    return ExecutionEngine(
        slack_gateway=slack,
        affected_services=affected_services or [],
    )


# ===========================================================================
# TestPolicyEnvelopeRules — unit tests per rule
# ===========================================================================


class TestPolicyEnvelopeRules:
    """Unit tests for each of the five PolicyEnvelope rules in isolation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.envelope = PolicyEnvelope()

    def _eval(self, action: RemediationAction, *, now: datetime = BUSINESS_HOURS_TS):
        plan = _plan(action)
        return self.envelope.evaluate(plan, now=now)[0]

    # ── PE-R1: Namespace Allowlist ───────────────────────────────────────

    def test_pe_r1_rejects_kube_system_namespace(self):
        """kube-system is not in the allowlist → REJECTED."""
        action = _action(namespace="kube-system")
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R1 for v in decision.violations)

    def test_pe_r1_rejects_monitoring_namespace(self):
        """monitoring is intentionally excluded → REJECTED."""
        action = _action(namespace="monitoring")
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R1 for v in decision.violations)

    def test_pe_r1_rejects_cert_manager_namespace(self):
        """cert-manager is not in allowlist → REJECTED."""
        action = _action(namespace="cert-manager")
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.REJECTED

    def test_pe_r1_allows_production_namespace(self):
        """production is in the allowlist → not rejected by R1."""
        action = _action(namespace="production")
        decision = self._eval(action)
        assert not any(v.rule_id == PE_R1 for v in decision.violations)

    def test_pe_r1_allows_staging_namespace(self):
        """staging is in the allowlist."""
        action = _action(namespace="staging")
        decision = self._eval(action)
        assert not any(v.rule_id == PE_R1 for v in decision.violations)

    def test_pe_r1_allows_default_namespace(self):
        """default is in the allowlist."""
        action = _action(namespace="default")
        decision = self._eval(action)
        assert not any(v.rule_id == PE_R1 for v in decision.violations)

    def test_pe_r1_violation_reason_mentions_namespace(self):
        """R1 violation reason must reference the offending namespace."""
        action = _action(namespace="kube-system")
        decision = self._eval(action)
        r1_violations = [v for v in decision.violations if v.rule_id == PE_R1]
        assert len(r1_violations) == 1
        assert "kube-system" in r1_violations[0].reason

    # ── PE-R2: Blast Radius Cap ──────────────────────────────────────────

    def test_pe_r2_rejects_blast_radius_51(self):
        """blast_radius=51 exceeds the LOW-risk cap of 50 → REJECTED."""
        action = _action(risk=RiskLevel.LOW, blast_radius=51)
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R2 for v in decision.violations)

    def test_pe_r2_allows_blast_radius_50(self):
        """blast_radius=50 is exactly at the cap → not rejected by R2."""
        action = _action(risk=RiskLevel.LOW, blast_radius=50)
        decision = self._eval(action)
        r2 = [v for v in decision.violations if v.rule_id == PE_R2]
        assert len(r2) == 0

    def test_pe_r2_critical_cap_is_10(self):
        """CRITICAL actions have a tighter cap of 10 pods."""
        action = _action(risk=RiskLevel.CRITICAL, blast_radius=11)
        decision = self._eval(action)
        assert any(v.rule_id == PE_R2 for v in decision.violations)

    def test_pe_r2_critical_allows_blast_radius_10(self):
        """blast_radius=10 is within CRITICAL cap → R2 passes."""
        action = _action(risk=RiskLevel.CRITICAL, blast_radius=10)
        decision = self._eval(action)
        r2 = [v for v in decision.violations if v.rule_id == PE_R2]
        assert len(r2) == 0

    def test_pe_r2_violation_reason_mentions_cap(self):
        """R2 reason must reference the cap and the actual radius."""
        action = _action(risk=RiskLevel.LOW, blast_radius=100)
        decision = self._eval(action)
        r2 = [v for v in decision.violations if v.rule_id == PE_R2]
        assert "100" in r2[0].reason
        assert "50" in r2[0].reason  # cap value

    # ── PE-R3: Business Hours ────────────────────────────────────────────

    def test_pe_r3_rejects_drain_node_outside_business_hours(self):
        """drain_node at 22:00 UTC Saturday → REJECTED."""
        action = _action(kind=ActionKind.DRAIN_NODE, resource="node-1")
        decision = self._eval(action, now=OUTSIDE_HOURS_TS)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R3 for v in decision.violations)

    def test_pe_r3_rejects_exec_script_outside_business_hours(self):
        """exec_script at 22:00 UTC Saturday → REJECTED (R3 fires before R5 routing)."""
        action = _action(kind=ActionKind.EXEC_SCRIPT, resource="cleanup.sh")
        decision = self._eval(action, now=OUTSIDE_HOURS_TS)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R3 for v in decision.violations)

    def test_pe_r3_rejects_drain_node_before_business_hours(self):
        """drain_node at 08:00 UTC Monday (before open) → REJECTED."""
        action = _action(kind=ActionKind.DRAIN_NODE, resource="node-1")
        decision = self._eval(action, now=BEFORE_HOURS_TS)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R3 for v in decision.violations)

    def test_pe_r3_rejects_at_exactly_close(self):
        """drain_node at exactly 17:00 UTC (exclusive end) → REJECTED."""
        action = _action(kind=ActionKind.DRAIN_NODE, resource="node-1")
        decision = self._eval(action, now=AT_CLOSE_TS)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R3 for v in decision.violations)

    def test_pe_r3_allows_drain_node_inside_business_hours(self):
        """drain_node at 10:30 UTC Monday → R3 passes."""
        action = _action(kind=ActionKind.DRAIN_NODE, resource="node-1", risk=RiskLevel.HIGH)
        decision = self._eval(action, now=BUSINESS_HOURS_TS)
        r3 = [v for v in decision.violations if v.rule_id == PE_R3]
        assert len(r3) == 0

    def test_pe_r3_allows_drain_node_at_16_59_utc(self):
        """drain_node at 16:59 UTC Friday (one minute before close) → R3 passes."""
        action = _action(kind=ActionKind.DRAIN_NODE, resource="node-1", risk=RiskLevel.HIGH)
        decision = self._eval(action, now=NEAR_CLOSE_TS)
        r3 = [v for v in decision.violations if v.rule_id == PE_R3]
        assert len(r3) == 0

    def test_pe_r3_emergency_flag_bypasses_hours_restriction(self):
        """emergency=True overrides the business-hours gate."""
        action = _action(
            kind=ActionKind.DRAIN_NODE,
            resource="node-1",
            risk=RiskLevel.HIGH,
            params={"emergency": True},
        )
        decision = self._eval(action, now=OUTSIDE_HOURS_TS)
        r3 = [v for v in decision.violations if v.rule_id == PE_R3]
        assert len(r3) == 0, "emergency=True must bypass PE-R3"

    def test_pe_r3_does_not_apply_to_restart_pod(self):
        """restart_pod is not in the restricted kinds → R3 never fires."""
        action = _action(kind=ActionKind.RESTART_POD, resource="api-gateway")
        decision = self._eval(action, now=OUTSIDE_HOURS_TS)
        r3 = [v for v in decision.violations if v.rule_id == PE_R3]
        assert len(r3) == 0

    # ── PE-R4: Risk Routing ──────────────────────────────────────────────

    def test_pe_r4_routes_high_risk_to_pending_human(self):
        """HIGH risk action → PENDING_HUMAN."""
        action = _action(risk=RiskLevel.HIGH)
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.PENDING_HUMAN
        assert decision.approved is False

    def test_pe_r4_rejects_critical_risk(self):
        """CRITICAL risk action → REJECTED."""
        action = _action(risk=RiskLevel.CRITICAL, blast_radius=1)
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.REJECTED
        assert any(v.rule_id == PE_R4 for v in decision.violations)

    def test_pe_r4_approves_low_risk(self):
        """LOW risk action (passing all other rules) → AUTO_APPROVED."""
        action = _action(risk=RiskLevel.LOW)
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.AUTO_APPROVED
        assert decision.approved is True

    def test_pe_r4_approves_medium_risk(self):
        """MEDIUM risk action → AUTO_APPROVED."""
        action = _action(risk=RiskLevel.MEDIUM)
        decision = self._eval(action)
        assert decision.decision == DecisionOutcome.AUTO_APPROVED

    def test_pe_r4_critical_rejection_has_rule_violation(self):
        """CRITICAL rejection must produce a PE-R4 violation."""
        action = _action(risk=RiskLevel.CRITICAL, blast_radius=1)
        decision = self._eval(action)
        r4 = [v for v in decision.violations if v.rule_id == PE_R4]
        assert len(r4) == 1

    # ── PE-R5: exec_script Always High ──────────────────────────────────

    def test_pe_r5_exec_script_low_declared_routed_to_slack(self):
        """exec_script with declared LOW risk → effective HIGH → PENDING_HUMAN."""
        action = _action(kind=ActionKind.EXEC_SCRIPT, risk=RiskLevel.LOW, resource="fix.sh")
        decision = self._eval(action)
        assert decision.effective_risk_level == RiskLevel.HIGH
        assert decision.decision == DecisionOutcome.PENDING_HUMAN

    def test_pe_r5_exec_script_medium_declared_routed_to_slack(self):
        """exec_script with declared MEDIUM risk → effective HIGH → PENDING_HUMAN."""
        action = _action(kind=ActionKind.EXEC_SCRIPT, risk=RiskLevel.MEDIUM, resource="fix.sh")
        decision = self._eval(action, now=BUSINESS_HOURS_TS)
        assert decision.effective_risk_level == RiskLevel.HIGH
        assert decision.decision == DecisionOutcome.PENDING_HUMAN

    def test_pe_r5_exec_script_high_declared_stays_high(self):
        """exec_script with declared HIGH → still HIGH (no change, correct outcome)."""
        action = _action(kind=ActionKind.EXEC_SCRIPT, risk=RiskLevel.HIGH, resource="fix.sh")
        decision = self._eval(action, now=BUSINESS_HOURS_TS)
        assert decision.effective_risk_level == RiskLevel.HIGH

    def test_pe_r5_does_not_affect_restart_pod(self):
        """restart_pod with LOW risk stays LOW → AUTO_APPROVED."""
        action = _action(kind=ActionKind.RESTART_POD, risk=RiskLevel.LOW)
        decision = self._eval(action)
        assert decision.effective_risk_level == RiskLevel.LOW
        assert decision.decision == DecisionOutcome.AUTO_APPROVED

    def test_pe_r5_effective_risk_reflected_in_decision(self):
        """PolicyDecision.effective_risk_level always reflects post-R5 risk."""
        action = _action(kind=ActionKind.EXEC_SCRIPT, risk=RiskLevel.LOW, resource="s.sh")
        decision = self._eval(action)
        assert decision.effective_risk_level == RiskLevel.HIGH
        assert decision.effective_risk_level != action.risk_level  # was upgraded


# ===========================================================================
# TestSafeRunbook — end-to-end safe runbook
# ===========================================================================


class TestSafeRunbook:
    """
    End-to-end tests for a compliant, low-risk remediation plan.

    Runbook: restart_pod on 'production' namespace, blast_radius=2, risk=LOW,
    evaluated during business hours.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.slack = MockSlackGateway()
        self.engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=["payments-service"],
        )

    def _safe_plan(self) -> RemediationPlan:
        return _plan(
            _action(
                kind=ActionKind.RESTART_POD,
                namespace="production",
                resource="payments-service",
                risk=RiskLevel.LOW,
                blast_radius=2,
                runbook_ref="RB-SAFE-001",
            )
        )

    @pytest.mark.asyncio
    async def test_safe_runbook_is_auto_approved(self):
        """Safe runbook → AUTO_APPROVED with the action in actions_approved."""
        result = await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True},
        )
        assert len(result.actions_approved) == 1
        assert len(result.actions_rejected) == 0
        assert len(result.actions_pending_human) == 0

    @pytest.mark.asyncio
    async def test_safe_runbook_does_not_route_to_slack(self):
        """Safe runbook must NOT trigger any Slack messages."""
        await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True},
        )
        assert not self.slack.was_called

    @pytest.mark.asyncio
    async def test_safe_runbook_no_rollback_when_healthy(self):
        """When all health checks pass, rollback must NOT be triggered."""
        result = await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True},
        )
        assert result.rollback_triggered is False
        assert result.rollback_reason == ""

    @pytest.mark.asyncio
    async def test_safe_runbook_rollback_triggered_on_health_failure(self):
        """
        When the approved service is unhealthy post-execution, the mandatory
        rollback trigger must fire.
        """
        result = await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": False},  # inject failure
        )
        assert result.rollback_triggered is True

    @pytest.mark.asyncio
    async def test_safe_runbook_rollback_reason_mentions_health_check(self):
        """The rollback_reason must contain the string 'health_check'."""
        result = await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": False},
        )
        assert "health_check" in result.rollback_reason
        assert "payments-service" in result.rollback_reason

    @pytest.mark.asyncio
    async def test_safe_runbook_actions_approved_contains_action_id_strings(self):
        """actions_approved must contain action_id strings, not code outputs."""
        plan = self._safe_plan()
        result = await self.engine.execute_plan(
            plan,
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True},
        )
        assert len(result.actions_approved) == 1
        action_id = plan.actions[0].action_id
        assert result.actions_approved[0] == action_id
        assert isinstance(result.actions_approved[0], str)

    @pytest.mark.asyncio
    async def test_safe_runbook_health_checks_are_recorded(self):
        """ExecutionResult must contain health check records for all probed services."""
        result = await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True},
        )
        assert len(result.health_checks) == 1
        assert result.health_checks[0].service == "payments-service"
        assert result.health_checks[0].healthy is True

    @pytest.mark.asyncio
    async def test_safe_runbook_policy_decision_in_result(self):
        """ExecutionResult must contain exactly one PolicyDecision for one action."""
        result = await self.engine.execute_plan(
            self._safe_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True},
        )
        assert len(result.policy_decisions) == 1
        assert result.policy_decisions[0].decision == DecisionOutcome.AUTO_APPROVED


# ===========================================================================
# TestUnsafeRunbook — end-to-end unsafe runbook
# ===========================================================================


class TestUnsafeRunbook:
    """
    End-to-end tests proving that non-compliant or high-risk actions are
    stopped deterministically by the policy engine.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.slack = MockSlackGateway()
        self.engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=[],
        )

    @pytest.mark.asyncio
    async def test_unsafe_exec_script_routed_to_slack(self):
        """
        exec_script (any declared risk) → PE-R5 upgrades to HIGH → PE-R4 routes
        to PENDING_HUMAN → Slack message sent.
        """
        plan = _plan(
            _action(
                kind=ActionKind.EXEC_SCRIPT,
                namespace="production",
                resource="cleanup.sh",
                risk=RiskLevel.LOW,   # deliberately underscored
            )
        )
        result = await self.engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        assert len(result.actions_pending_human) == 1
        assert len(result.actions_approved) == 0
        assert self.slack.was_called

    @pytest.mark.asyncio
    async def test_unsafe_exec_script_slack_message_contains_action_id(self):
        """Slack message must reference the action_id for audit traceability."""
        plan = _plan(
            _action(kind=ActionKind.EXEC_SCRIPT, namespace="production", resource="s.sh")
        )
        action_id = plan.actions[0].action_id
        await self.engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        msgs = self.slack.messages_for_action(action_id)
        assert len(msgs) == 1
        assert msgs[0]["action_id"] == action_id

    @pytest.mark.asyncio
    async def test_unsafe_forbidden_namespace_rejected(self):
        """kube-system namespace → PE-R1 → REJECTED, no Slack message."""
        plan = _plan(
            _action(namespace="kube-system", resource="coredns")
        )
        result = await self.engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        assert len(result.actions_rejected) == 1
        assert len(result.actions_approved) == 0
        assert not self.slack.was_called

    @pytest.mark.asyncio
    async def test_unsafe_monitoring_namespace_rejected(self):
        """monitoring namespace → PE-R1 → REJECTED (observability stack protected)."""
        plan = _plan(
            _action(namespace="monitoring", resource="prometheus-server")
        )
        result = await self.engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        assert len(result.actions_rejected) == 1
        assert not self.slack.was_called

    @pytest.mark.asyncio
    async def test_unsafe_drain_node_outside_hours_rejected(self):
        """drain_node at 22:00 UTC Saturday, no emergency flag → REJECTED."""
        plan = _plan(
            _action(
                kind=ActionKind.DRAIN_NODE,
                namespace="production",
                resource="node-1",
                risk=RiskLevel.MEDIUM,
            )
        )
        result = await self.engine.execute_plan(plan, now=OUTSIDE_HOURS_TS)
        assert len(result.actions_rejected) == 1
        # Verify PE-R3 specifically fired
        d = result.policy_decisions[0]
        assert any(v.rule_id == PE_R3 for v in d.violations)

    @pytest.mark.asyncio
    async def test_unsafe_critical_risk_always_rejected(self):
        """CRITICAL risk → REJECTED, regardless of namespace or hours."""
        plan = _plan(
            _action(
                namespace="production",
                risk=RiskLevel.CRITICAL,
                blast_radius=1,  # even tiny blast radius can't save CRITICAL
            )
        )
        result = await self.engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        assert len(result.actions_rejected) == 1
        assert len(result.actions_approved) == 0
        assert not self.slack.was_called

    @pytest.mark.asyncio
    async def test_unsafe_large_blast_radius_rejected(self):
        """blast_radius=200 exceeds cap → PE-R2 → REJECTED."""
        plan = _plan(
            _action(namespace="production", risk=RiskLevel.LOW, blast_radius=200)
        )
        result = await self.engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        assert len(result.actions_rejected) == 1
        d = result.policy_decisions[0]
        assert any(v.rule_id == PE_R2 for v in d.violations)

    @pytest.mark.asyncio
    async def test_unsafe_mixed_plan_per_action_gating(self):
        """
        Critical test: a mixed plan with one safe action and one high-risk
        exec_script must be gated PER ACTION, not at the plan level.

        - restart_pod / LOW / production → AUTO_APPROVED
        - exec_script / LOW / production → PE-R5 upgrades → PENDING_HUMAN
        """
        restart_action = _action(
            kind=ActionKind.RESTART_POD,
            namespace="production",
            resource="payments-service",
            risk=RiskLevel.LOW,
            blast_radius=2,
        )
        script_action = _action(
            kind=ActionKind.EXEC_SCRIPT,
            namespace="production",
            resource="cleanup.sh",
            risk=RiskLevel.LOW,   # declared low but PE-R5 upgrades
        )
        plan = _plan(restart_action, script_action)
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=["payments-service"],
        )
        result = await engine.execute_plan(plan, now=BUSINESS_HOURS_TS)

        assert len(result.actions_approved) == 1
        assert result.actions_approved[0] == restart_action.action_id

        assert len(result.actions_pending_human) == 1
        assert result.actions_pending_human[0] == script_action.action_id

        assert len(result.actions_rejected) == 0
        assert self.slack.was_called

    @pytest.mark.asyncio
    async def test_unsafe_mixed_plan_slack_only_for_pending(self):
        """Slack is called exactly once — for the PENDING_HUMAN action, not the approved one."""
        restart_action = _action(
            kind=ActionKind.RESTART_POD, namespace="production",
            resource="api-gw", risk=RiskLevel.LOW
        )
        script_action = _action(
            kind=ActionKind.EXEC_SCRIPT, namespace="production",
            resource="s.sh", risk=RiskLevel.LOW
        )
        plan = _plan(restart_action, script_action)
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=["api-gw"],
        )
        await engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        # Only one Slack message — not two
        assert len(self.slack.sent_messages) == 1
        # It must be for the exec_script action
        assert self.slack.sent_messages[0]["action_id"] == script_action.action_id


# ===========================================================================
# TestRollbackTrigger — mandatory rollback invariant
# ===========================================================================


class TestRollbackTrigger:
    """Proves the mandatory rollback invariant in all combinations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = ExecutionEngine(
            affected_services=["svc-a", "svc-b"],
        )

    def _simple_plan(self) -> RemediationPlan:
        return _plan(
            _action(
                kind=ActionKind.RESTART_POD,
                namespace="production",
                resource="svc-a",
                risk=RiskLevel.LOW,
                blast_radius=1,
            )
        )

    @pytest.mark.asyncio
    async def test_rollback_triggered_when_one_service_unhealthy(self):
        """If any service is unhealthy, rollback_triggered must be True."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": False, "svc-b": True},
        )
        assert result.rollback_triggered is True

    @pytest.mark.asyncio
    async def test_rollback_triggered_when_all_services_unhealthy(self):
        """All unhealthy → rollback_triggered=True."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": False, "svc-b": False},
        )
        assert result.rollback_triggered is True

    @pytest.mark.asyncio
    async def test_rollback_not_triggered_when_all_healthy(self):
        """All healthy → rollback_triggered=False."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": True, "svc-b": True},
        )
        assert result.rollback_triggered is False

    @pytest.mark.asyncio
    async def test_rollback_reason_contains_health_check_failed(self):
        """rollback_reason must contain the machine-readable marker."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": False, "svc-b": True},
        )
        assert "health_check_failed" in result.rollback_reason

    @pytest.mark.asyncio
    async def test_rollback_reason_names_unhealthy_services(self):
        """rollback_reason must name the unhealthy service(s)."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": False, "svc-b": True},
        )
        assert "svc-a" in result.rollback_reason

    @pytest.mark.asyncio
    async def test_rollback_reason_empty_when_not_triggered(self):
        """rollback_reason must be empty string when rollback is not triggered."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": True, "svc-b": True},
        )
        assert result.rollback_reason == ""

    @pytest.mark.asyncio
    async def test_health_check_result_recorded_for_unhealthy_service(self):
        """The unhealthy HealthCheckResult must be present in health_checks."""
        result = await self.engine.execute_plan(
            self._simple_plan(),
            now=BUSINESS_HOURS_TS,
            health_overrides={"svc-a": False, "svc-b": True},
        )
        unhealthy = [hc for hc in result.health_checks if not hc.healthy]
        assert len(unhealthy) == 1
        assert unhealthy[0].service == "svc-a"
        assert unhealthy[0].http_status == 503


# ===========================================================================
# TestIronInvariants — cross-cutting invariant proofs
# ===========================================================================


class TestIronInvariants:
    """
    Cross-cutting invariants that must hold across all scenarios.
    These are the five 'iron' guarantees of the policy layer.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.slack = MockSlackGateway()

    # ── Invariant 1: CRITICAL risk → always REJECTED ──────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize("namespace,blast_radius,now_ts", [
        ("production", 1, BUSINESS_HOURS_TS),
        ("staging", 5, BUSINESS_HOURS_TS),
        ("default", 10, BUSINESS_HOURS_TS),
        ("production", 10, OUTSIDE_HOURS_TS),  # outside hours too
    ])
    async def test_invariant_critical_always_rejected(
        self, namespace, blast_radius, now_ts
    ):
        """
        CRITICAL risk is always REJECTED regardless of namespace, blast radius,
        or time of day.  There is no approval path for CRITICAL actions.
        """
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=[],
        )
        plan = _plan(
            _action(
                namespace=namespace,
                risk=RiskLevel.CRITICAL,
                blast_radius=blast_radius,
            )
        )
        result = await engine.execute_plan(plan, now=now_ts)
        assert result.actions_rejected == [plan.actions[0].action_id], (
            f"INVARIANT BROKEN: CRITICAL action in ns={namespace} br={blast_radius} "
            f"was not REJECTED. Got: {result.policy_decisions[0].decision}"
        )
        assert len(result.actions_approved) == 0

    # ── Invariant 2: exec_script → always PENDING_HUMAN (or REJECTED if other rules fire)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("declared_risk", [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH])
    async def test_invariant_exec_script_never_auto_approved(self, declared_risk):
        """
        exec_script with any declared risk ≤ HIGH must NEVER be AUTO_APPROVED.
        PE-R5 always upgrades it to HIGH; PE-R4 then routes to PENDING_HUMAN.
        """
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=[],
        )
        plan = _plan(
            _action(
                kind=ActionKind.EXEC_SCRIPT,
                namespace="production",
                resource="script.sh",
                risk=declared_risk,
            )
        )
        result = await engine.execute_plan(plan, now=BUSINESS_HOURS_TS)
        assert plan.actions[0].action_id not in result.actions_approved, (
            f"INVARIANT BROKEN: exec_script with declared risk={declared_risk.value} "
            f"was AUTO_APPROVED. This must never happen."
        )

    # ── Invariant 3: Rollback fires if and only if any_unhealthy ─────

    @pytest.mark.asyncio
    @pytest.mark.parametrize("health_map,expect_rollback", [
        ({"svc": True}, False),
        ({"svc": False}, True),
        ({"svc-a": True, "svc-b": False}, True),
        ({"svc-a": True, "svc-b": True}, False),
    ])
    async def test_invariant_rollback_iff_unhealthy(self, health_map, expect_rollback):
        """Rollback is triggered if and only if any service is unhealthy."""
        affected = list(health_map.keys())
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=affected,
        )
        plan = _plan(
            _action(namespace="production", resource=affected[0], risk=RiskLevel.LOW)
        )
        result = await engine.execute_plan(
            plan, now=BUSINESS_HOURS_TS, health_overrides=health_map
        )
        assert result.rollback_triggered == expect_rollback, (
            f"INVARIANT BROKEN: health_map={health_map}, "
            f"expected rollback={expect_rollback}, got {result.rollback_triggered}"
        )

    # ── Invariant 4: Slack called iff and only iff PENDING_HUMAN ──────

    @pytest.mark.asyncio
    async def test_invariant_slack_called_iff_pending_human(self):
        """
        Slack must be called exactly once per PENDING_HUMAN decision
        and never for AUTO_APPROVED or REJECTED decisions.
        """
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=[],
        )
        # Three actions: one approved, one pending, one rejected
        approved_action = _action(
            kind=ActionKind.RESTART_POD, namespace="production",
            resource="svc-a", risk=RiskLevel.LOW
        )
        pending_action = _action(
            kind=ActionKind.RESTART_POD, namespace="production",
            resource="svc-b", risk=RiskLevel.HIGH  # HIGH → PENDING_HUMAN
        )
        rejected_action = _action(
            kind=ActionKind.RESTART_POD, namespace="kube-system",  # forbidden ns
            resource="svc-c", risk=RiskLevel.LOW
        )
        plan = _plan(approved_action, pending_action, rejected_action)
        result = await engine.execute_plan(plan, now=BUSINESS_HOURS_TS)

        assert len(result.actions_approved) == 1
        assert len(result.actions_pending_human) == 1
        assert len(result.actions_rejected) == 1

        # Exactly one Slack message — only for the pending action
        assert len(self.slack.sent_messages) == 1
        assert self.slack.sent_messages[0]["action_id"] == pending_action.action_id

    # ── Invariant 5: actions_approved contains only ID strings ────────

    @pytest.mark.asyncio
    async def test_invariant_actions_approved_contains_only_id_strings(self):
        """
        actions_approved must be a list of action_id strings.
        The engine must never put code outputs, execution results,
        or callable objects in this list.
        """
        engine = ExecutionEngine(
            slack_gateway=self.slack,
            affected_services=["payments-service"],
        )
        plan = _plan(
            _action(namespace="production", resource="payments-service", risk=RiskLevel.LOW),
            _action(namespace="production", resource="api-gateway", risk=RiskLevel.LOW),
        )
        result = await engine.execute_plan(
            plan, now=BUSINESS_HOURS_TS,
            health_overrides={"payments-service": True, "api-gateway": True},
        )
        for item in result.actions_approved:
            assert isinstance(item, str), f"Expected str, got {type(item)}: {item!r}"
            # Must be a UUID-formatted action_id, not a command string
            assert not any(
                keyword in item
                for keyword in ["kubectl", "exec", "bash", "sh", "python", "sudo"]
            ), f"actions_approved contains command-like string: {item!r}"
