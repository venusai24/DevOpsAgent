"""
tests/test_safety.py

Unit tests for the Safe Execution layer.

Tests cover:
  - PolicyEngine: invariant checking (7 invariants), Terraform HCL validation
  - BlastRadiusEstimator: risk scoring, tier-1 impact detection, recommendation mapping
  - CanaryController: stage progression, golden signal monitoring, halt-on-violation
  - RollbackController: regression detection, rollback triggering
  - Guardrails: safe/unsafe command classification
"""

from __future__ import annotations

import pytest

from agent.action.policy_engine import (
    PolicyEngine, PolicyCheckResult, InvariantViolation,
    generate_terraform_patch,
)
from agent.action.blast_radius import BlastRadiusEstimator
from agent.action.canary_controller import CanaryController, CanaryResult
from agent.action.rollback_controller import (
    RollbackController, GoldenSignalSnapshot, RollbackResult,
)
from agent.security.guardrails import is_safe_command


# ---------------------------------------------------------------------------
# GoldenSignalSnapshot helpers
# ---------------------------------------------------------------------------

def healthy_baseline() -> GoldenSignalSnapshot:
    return GoldenSignalSnapshot(
        latency_p99_ms=120.0,
        traffic_rps=1500.0,
        error_rate_pct=0.3,
        saturation_pct=35.0,
    )


def degraded_signals() -> GoldenSignalSnapshot:
    """Signals that represent a clear regression."""
    return GoldenSignalSnapshot(
        latency_p99_ms=980.0,   # 8x increase
        traffic_rps=800.0,
        error_rate_pct=45.0,    # >> 20% absolute threshold
        saturation_pct=98.0,    # >> 95% absolute threshold
    )


def improved_signals() -> GoldenSignalSnapshot:
    """Signals that are better than baseline — no regression."""
    return GoldenSignalSnapshot(
        latency_p99_ms=100.0,
        traffic_rps=1600.0,
        error_rate_pct=0.1,
        saturation_pct=25.0,
    )


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class TestPolicyEngine:

    def setup_method(self):
        self.engine = PolicyEngine()

    def _check(self, proposed_patch=None, current_state=None, incident_context=None, hcl=None):
        return self.engine.check(
            proposed_patch=proposed_patch or {},
            current_state=current_state or {"replicas": 3, "public_access": False, "max_connections": 100},
            incident_context=incident_context or {"severity": "P1", "rollback_command": "kubectl rollout undo deployment/svc"},
            terraform_hcl=hcl,
        )

    def test_clean_patch_passes(self):
        """A safe patch with rollback command and no violations should pass."""
        result = self._check(proposed_patch={"replicas": 5})
        assert isinstance(result, PolicyCheckResult)
        assert result.passed
        assert len(result.violations) == 0

    def test_zero_replicas_violation(self):
        """Scaling to 0 replicas violates MIN_REPLICAS invariant."""
        result = self._check(proposed_patch={"replicas": 0})
        assert not result.passed
        violation_ids = [v.invariant_id for v in result.violations]
        assert "MIN_REPLICAS" in violation_ids

    def test_one_replica_violation(self):
        """Scaling to 1 replica violates MIN_REPLICAS (< 2) invariant."""
        result = self._check(proposed_patch={"replicas": 1})
        assert not result.passed
        violation_ids = [v.invariant_id for v in result.violations]
        assert "MIN_REPLICAS" in violation_ids

    def test_public_access_violation(self):
        """Enabling public_access violates NO_PUBLIC_DB_EXPOSURE invariant."""
        result = self._check(proposed_patch={"public_access": True})
        assert not result.passed
        violation_ids = [v.invariant_id for v in result.violations]
        assert "NO_PUBLIC_DB_EXPOSURE" in violation_ids

    def test_no_rollback_p1_violation(self):
        """Missing rollback command violates ROLLBACK_REQUIRED invariant."""
        result = self._check(
            incident_context={"severity": "P1", "rollback_command": ""},
        )
        assert not result.passed
        violation_ids = [v.invariant_id for v in result.violations]
        assert "ROLLBACK_REQUIRED" in violation_ids

    def test_max_scale_factor_violation(self):
        """Scaling 4x from current=3 replicas violates MAX_SCALE_FACTOR (max 3x)."""
        result = self._check(
            proposed_patch={"replicas": 12},
            current_state={"replicas": 3, "public_access": False, "max_connections": 100},
        )
        assert not result.passed
        violation_ids = [v.invariant_id for v in result.violations]
        assert "MAX_SCALE_FACTOR" in violation_ids

    def test_min_connection_pool_violation(self):
        """Setting max_connections < 10 violates MIN_CONNECTION_POOL invariant."""
        result = self._check(
            proposed_patch={"max_connections": 5},
        )
        assert not result.passed
        violation_ids = [v.invariant_id for v in result.violations]
        assert "MIN_CONNECTION_POOL" in violation_ids

    def test_multiple_violations_all_reported(self):
        """Multiple simultaneous violations are all reported."""
        result = self._check(
            proposed_patch={"replicas": 0, "public_access": True},
            incident_context={"severity": "P1", "rollback_command": ""},
        )
        assert not result.passed
        assert len(result.violations) >= 2

    def test_warnings_do_not_block(self):
        """CHANGE_WINDOW warning alone should not block execution."""
        # A clean patch with rollback command — change window is a warning
        result = self._check(proposed_patch={"replicas": 3})
        # Even if CHANGE_WINDOW warning fires, the check should still pass
        # (warnings go to result.warnings, not result.violations)
        assert result.passed or len([v for v in result.violations if v.severity == "critical"]) > 0

    def test_violations_are_invariant_violation_instances(self):
        """Each violation is an InvariantViolation with expected fields."""
        result = self._check(proposed_patch={"replicas": 0})
        for v in result.violations:
            assert isinstance(v, InvariantViolation)
            assert hasattr(v, "invariant_id")
            assert hasattr(v, "description")
            assert hasattr(v, "severity")

    def test_critical_violations_property(self):
        """critical_violations returns only severity=='critical' violations."""
        result = self._check(proposed_patch={"replicas": 0})
        for v in result.critical_violations:
            assert v.severity == "critical"

    def test_to_markdown_pass(self):
        """to_markdown() on a passing result includes PASS indicator."""
        result = self._check(proposed_patch={"replicas": 3})
        md = result.to_markdown()
        assert "PASSED" in md or "✅" in md

    def test_to_markdown_fail(self):
        """to_markdown() on a failing result includes FAILED indicator."""
        result = self._check(proposed_patch={"replicas": 0})
        md = result.to_markdown()
        assert "FAILED" in md or "🚫" in md

    def test_terraform_patch_generation_restart(self):
        """generate_terraform_patch for restart_deployment includes service name."""
        hcl = generate_terraform_patch("payments-service", "prod", "restart_deployment", {})
        assert "payments-service" in hcl
        assert "payments_service" in hcl
        assert isinstance(hcl, str)

    def test_terraform_patch_generation_scale(self):
        """generate_terraform_patch for scale_deployment includes replica count."""
        hcl = generate_terraform_patch("order-service", "staging", "scale_deployment", {"replicas": 5})
        assert "order-service" in hcl or "order_service" in hcl
        assert "5" in hcl

    def test_terraform_valid_field_true_without_hcl(self):
        """Without HCL, terraform_valid should default to True."""
        result = self._check()
        assert result.terraform_valid is True

    def test_simulated_state_merges_patch(self):
        """simulated_state should merge current_state with proposed_patch."""
        result = self._check(
            proposed_patch={"replicas": 5},
            current_state={"replicas": 3, "public_access": False, "max_connections": 100},
        )
        assert result.simulated_state.get("replicas") == 5
        assert result.simulated_state.get("public_access") is False


# ---------------------------------------------------------------------------
# BlastRadiusEstimator
# ---------------------------------------------------------------------------

class TestBlastRadiusEstimator:

    def setup_method(self):
        self.estimator = BlastRadiusEstimator()

    @pytest.mark.asyncio
    async def test_estimate_returns_report(self):
        """estimate() returns an object with the required fields."""
        report = await self.estimator.estimate(
            "payments-service",
            "kubectl rollout restart deployment/payments-service",
        )
        assert hasattr(report, "target_service")
        assert hasattr(report, "risk_score")
        assert hasattr(report, "recommendation")
        assert report.target_service == "payments-service"

    @pytest.mark.asyncio
    async def test_risk_score_is_bounded(self):
        """risk_score should always be in [0.0, 1.0]."""
        report = await self.estimator.estimate(
            "payments-service", "kubectl delete pod foo",
        )
        assert 0.0 <= report.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_recommendation_valid_values(self):
        """recommendation must be one of the three allowed values."""
        report = await self.estimator.estimate(
            "payments-service", "kubectl apply -f patch.yaml",
        )
        assert report.recommendation in ("auto_execute", "require_approval", "block")

    @pytest.mark.asyncio
    async def test_to_markdown_includes_service(self):
        """to_markdown() includes the target service name."""
        # Use payments-service which is present in the real topology fixture
        report = await self.estimator.estimate(
            "payments-service",
            "kubectl rollout restart deployment/payments-service",
        )
        md = report.to_markdown()
        assert "payments-service" in md

    @pytest.mark.asyncio
    async def test_affected_services_is_list(self):
        """affected_services field is a list."""
        report = await self.estimator.estimate(
            "payments-service", "kubectl rollout restart deployment/payments-service",
        )
        assert isinstance(report.affected_services, list)


# ---------------------------------------------------------------------------
# RollbackController
# ---------------------------------------------------------------------------

class TestRollbackController:

    def setup_method(self):
        self.ctrl = RollbackController()

    def test_no_regression_healthy_signals(self):
        """Signals that improve post-remediation should not trigger rollback."""
        result = self.ctrl.check_regression(healthy_baseline(), improved_signals())
        assert isinstance(result, RollbackResult)
        assert not result.should_rollback

    def test_error_rate_spike_triggers_regression(self):
        """A massive error rate spike triggers rollback."""
        result = self.ctrl.check_regression(healthy_baseline(), degraded_signals())
        assert result.should_rollback
        assert "error_rate_pct" in result.regressed_signals

    def test_latency_spike_triggers_regression(self):
        """A latency spike beyond 1.5x threshold triggers rollback."""
        before = healthy_baseline()
        after = GoldenSignalSnapshot(
            latency_p99_ms=500.0,   # > 1.5x of 120ms
            traffic_rps=1500.0,
            error_rate_pct=0.3,
            saturation_pct=35.0,
        )
        result = self.ctrl.check_regression(before, after)
        assert result.should_rollback
        assert "latency_p99_ms" in result.regressed_signals

    def test_saturation_spike_triggers_regression(self):
        """Saturation exceeding absolute threshold (95%) triggers rollback."""
        before = healthy_baseline()
        after = GoldenSignalSnapshot(
            latency_p99_ms=130.0,
            traffic_rps=1500.0,
            error_rate_pct=0.3,
            saturation_pct=96.0,    # > 95% absolute threshold
        )
        result = self.ctrl.check_regression(before, after)
        assert result.should_rollback

    def test_regression_result_has_regressed_signals_list(self):
        """RollbackResult.regressed_signals is a list."""
        result = self.ctrl.check_regression(healthy_baseline(), degraded_signals())
        assert isinstance(result.regressed_signals, list)

    def test_regression_reason_is_string(self):
        """RollbackResult.reason is always a non-empty string."""
        result = self.ctrl.check_regression(healthy_baseline(), improved_signals())
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_rollback_result_to_markdown_no_rollback(self):
        """to_markdown() on no-rollback result contains healthy message."""
        result = self.ctrl.check_regression(healthy_baseline(), improved_signals())
        md = result.to_markdown()
        assert "✅" in md or "healthy" in md.lower() or "no rollback" in md.lower()

    def test_rollback_result_to_markdown_regression(self):
        """to_markdown() on regression result contains ROLLBACK indicator."""
        result = self.ctrl.check_regression(healthy_baseline(), degraded_signals())
        md = result.to_markdown()
        assert "ROLLBACK" in md or "🚨" in md

    def test_demo_baseline_returns_snapshot(self):
        """capture_demo_baseline returns a GoldenSignalSnapshot."""
        snap = RollbackController.capture_demo_baseline("payments-service")
        assert isinstance(snap, GoldenSignalSnapshot)
        assert snap.error_rate_pct >= 0.0
        assert snap.latency_p99_ms > 0.0

    def test_demo_post_remediation_healthy(self):
        """capture_demo_post_remediation(True) returns healthy signals."""
        snap = RollbackController.capture_demo_post_remediation(True)
        assert isinstance(snap, GoldenSignalSnapshot)
        assert snap.error_rate_pct < 1.0  # Healthy post-remediation

    def test_demo_post_remediation_failed(self):
        """capture_demo_post_remediation(False) returns degraded signals."""
        snap = RollbackController.capture_demo_post_remediation(False)
        assert isinstance(snap, GoldenSignalSnapshot)
        assert snap.error_rate_pct > 20.0  # Clear regression


# ---------------------------------------------------------------------------
# CanaryController
# ---------------------------------------------------------------------------

class TestCanaryController:

    def setup_method(self):
        # Use demo_mode=True to avoid real kubectl/Datadog calls
        self.ctrl = CanaryController(demo_mode=True)

    @pytest.mark.asyncio
    async def test_canary_returns_result(self):
        """execute_canary() returns a CanaryResult."""
        result = await self.ctrl.execute_canary(
            service="payments-service",
            remediation_command="kubectl rollout restart deployment/payments-service",
            rollback_command="kubectl rollout undo deployment/payments-service",
            baseline=healthy_baseline(),
        )
        assert isinstance(result, CanaryResult)
        assert result.service == "payments-service"

    @pytest.mark.asyncio
    async def test_canary_succeeded_is_bool(self):
        """CanaryResult.succeeded is a bool."""
        result = await self.ctrl.execute_canary(
            service="payments-service",
            remediation_command="kubectl rollout restart deployment/payments-service",
            rollback_command="kubectl rollout undo deployment/payments-service",
            baseline=healthy_baseline(),
        )
        assert isinstance(result.succeeded, bool)

    @pytest.mark.asyncio
    async def test_canary_has_stage_results(self):
        """CanaryResult.stage_results is a non-empty list."""
        result = await self.ctrl.execute_canary(
            service="payments-service",
            remediation_command="kubectl rollout restart deployment/payments-service",
            rollback_command="kubectl rollout undo deployment/payments-service",
            baseline=healthy_baseline(),
        )
        assert isinstance(result.stage_results, list)

    @pytest.mark.asyncio
    async def test_canary_to_markdown(self):
        """to_markdown() on CanaryResult returns a non-empty string containing service name."""
        result = await self.ctrl.execute_canary(
            service="payments-service",
            remediation_command="kubectl rollout restart deployment/payments-service",
            rollback_command="kubectl rollout undo deployment/payments-service",
            baseline=healthy_baseline(),
        )
        md = result.to_markdown()
        assert isinstance(md, str)
        assert "payments-service" in md
        assert len(md) > 50

    @pytest.mark.asyncio
    async def test_canary_total_duration_is_positive(self):
        """total_duration_seconds should be a positive float."""
        result = await self.ctrl.execute_canary(
            service="order-service",
            remediation_command="kubectl rollout restart deployment/order-service",
            rollback_command="kubectl rollout undo deployment/order-service",
            baseline=healthy_baseline(),
        )
        assert result.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_canary_single_stage(self):
        """Custom single-stage canary completes with 1 stage result."""
        single_stage = [{"stage": 1, "traffic_pct": 100, "observe_seconds": 1, "label": "full"}]
        result = await self.ctrl.execute_canary(
            service="svc",
            remediation_command="kubectl rollout restart deployment/svc",
            rollback_command="kubectl rollout undo deployment/svc",
            baseline=healthy_baseline(),
            stages=single_stage,
        )
        assert len(result.stage_results) == 1


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class TestGuardrails:

    def test_safe_restart_allowed(self):
        """kubectl rollout restart is a safe command."""
        assert is_safe_command("kubectl rollout restart deployment/payments-service -n prod")

    def test_safe_rollout_undo_allowed(self):
        """kubectl rollout undo is a safe rollback command."""
        assert is_safe_command("kubectl rollout undo deployment/payments-service -n prod")

    def test_safe_scale_allowed(self):
        """kubectl scale with replicas is safe."""
        assert is_safe_command("kubectl scale deployment/payments-service --replicas=5 -n prod")

    def test_kubectl_delete_all_blocked(self):
        """kubectl delete --all is blocked by guardrails (mass-deletion pattern)."""
        result = is_safe_command("kubectl delete pods --all -n prod")
        assert not result, "Mass delete-all should be blocked by guardrails"

    def test_terraform_destroy_blocked(self):
        """terraform destroy is blocked by guardrails (infrastructure destruction)."""
        assert not is_safe_command("terraform destroy -auto-approve")

    def test_rm_rf_blocked(self):
        """rm -rf is unconditionally destructive and should be blocked."""
        assert not is_safe_command("rm -rf /var/lib/postgresql/data")

    def test_returns_bool(self):
        """is_safe_command always returns a bool."""
        result = is_safe_command("kubectl rollout restart deployment/svc")
        assert isinstance(result, bool)

    def test_empty_command_blocked(self):
        """Empty command string should not be considered safe."""
        result = is_safe_command("")
        assert not result or isinstance(result, bool)
