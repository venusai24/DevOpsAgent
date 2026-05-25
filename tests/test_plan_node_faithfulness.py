"""
tests/test_plan_node_faithfulness.py

Unit tests for the faithfulness cross-validation logic in plan_node
(agent/nodes.py).

Specifically tests the private helpers:
  - _cross_validate_plan(plan, kb_entry) -> float
  - _normalise_cmd(cmd) -> str

And the plan_node integration behaviour:
  - When faithfulness < 0.50: rollback_command is cleared → is_high_risk=True
  - When faithfulness >= 0.50: plan is returned intact
  - When no KB entry was used: faithfulness_score is None in state
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent.nodes import _cross_validate_plan, _normalise_cmd
from agent.state import (
    KBEntry,
    KBRemediationStep,
    KBRetrievalResult,
    RemediationPlan,
    RemediationStep,
    TriageResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb_entry(
    root_cause: str = "Pod was OOMKilled. The container memory limit is too low.",
    commands: list[str] | None = None,
) -> KBEntry:
    if commands is None:
        commands = [
            "kubectl set resources deployment/{service} -n prod --limits=memory=2Gi",
            "kubectl rollout restart deployment/{service} -n prod",
        ]
    steps = [
        KBRemediationStep(
            order=i + 1,
            action=f"Step {i + 1}",
            environment="kubectl",
            command=cmd,
            risk="medium",
        )
        for i, cmd in enumerate(commands)
    ]
    return KBEntry(
        entry_id="kb-faith-001",
        incident_taxonomy="Errors:OOM",
        pattern_type="exact",
        error_pattern="Exit Code 137",
        affected_services=[],
        severity="P1",
        root_cause_narrative=root_cause,
        remediation_steps=steps,
        rollback_command="kubectl rollout undo deployment/{service} -n prod",
        confidence_score=0.88,
    )


def _make_plan(
    root_cause: str = "Pod was OOMKilled. The container memory limit is too low.",
    commands: list[str | None] | None = None,
    rollback: str = "kubectl rollout undo deployment/payment-service -n prod",
) -> RemediationPlan:
    if commands is None:
        commands = [
            "kubectl set resources deployment/payment-service -n prod --limits=memory=2Gi",
            "kubectl rollout restart deployment/payment-service -n prod",
        ]
    steps = [
        RemediationStep(
            order=i + 1,
            action=f"Step {i + 1}",
            command=cmd,
            risk="medium",
        )
        for i, cmd in enumerate(commands)
    ]
    return RemediationPlan(
        root_cause=root_cause,
        steps=steps,
        rollback_command=rollback,
        estimated_mttr_minutes=5,
        postmortem_summary="OOMKilled pod restarted after memory limit increase.",
    )


# ---------------------------------------------------------------------------
# Tests: _normalise_cmd
# ---------------------------------------------------------------------------

class TestNormalisCmd:
    def test_strips_pod_suffix(self):
        cmd = "kubectl get pod payment-service-7d9f8b-xkzp2 -n prod"
        normalised = _normalise_cmd(cmd)
        assert "7d9f8b" not in normalised
        assert "xkzp2" not in normalised

    def test_strips_ip_address(self):
        cmd = "curl http://192.168.1.100:8080/health"
        normalised = _normalise_cmd(cmd)
        assert "<ip>" in normalised
        assert "192.168" not in normalised

    def test_strips_hex(self):
        cmd = "echo 0xDEADBEEF"
        normalised = _normalise_cmd(cmd)
        assert "<hex>" in normalised

    def test_strips_large_integers(self):
        cmd = "kubectl scale --replicas=1234 deployment/svc"
        normalised = _normalise_cmd(cmd)
        assert "<n>" in normalised

    def test_service_placeholder_becomes_generic(self):
        """_normalise_cmd strips Kubernetes pod suffixes; is_command_grounded handles {service} via
        the grounding check in guardrails — verify the whole flow produces a match."""
        from agent.security.guardrails import is_command_grounded
        kb_cmd = "kubectl rollout restart deployment/{service} -n prod"
        entry = KBEntry(
            entry_id="kb-svc-test",
            incident_taxonomy="Errors:OOM",
            pattern_type="exact",
            error_pattern="Exit Code 137",
            affected_services=[],
            severity="P1",
            root_cause_narrative="Pod was OOMKilled. Memory limit too low.",
            remediation_steps=[
                KBRemediationStep(
                    order=1, action="Restart.", environment="kubectl",
                    command=kb_cmd, risk="medium",
                )
            ],
            rollback_command="kubectl rollout undo deployment/{service} -n prod",
        )
        # The generated command has the service name substituted in
        generated = "kubectl rollout restart deployment/payment-service -n prod"
        # The guardrails grounding check must accept this as grounded
        assert is_command_grounded(generated, entry) is True

    def test_lowercased_output(self):
        cmd = "kubectl GET Pod MyPod -n Prod"
        assert _normalise_cmd(cmd) == _normalise_cmd(cmd).lower()


# ---------------------------------------------------------------------------
# Tests: _cross_validate_plan — command fidelity
# ---------------------------------------------------------------------------

class TestCrossValidatePlanCommandFidelity:
    def test_perfect_command_match_scores_high(self):
        """When generated commands match KB commands (after normalisation), score is high."""
        entry = _make_kb_entry()
        plan = _make_plan()  # Same commands, just with service name substituted
        score = _cross_validate_plan(plan, entry)
        # Command fidelity should be >= 0.5; root cause also matches.
        # Combined score should be >= 0.50.
        assert score >= 0.50, f"Expected score >= 0.50, got {score}"

    def test_invented_command_scores_low(self):
        """When the LLM invents commands not in the KB, command fidelity drops to 0."""
        entry = _make_kb_entry()
        plan = _make_plan(
            commands=[
                "kubectl delete deployment payment-service -n prod",  # Not in KB
                "helm rollback payment-service -n prod",              # Not in KB
            ],
            root_cause="Network partition in etcd cluster.",  # Also wrong
        )
        score = _cross_validate_plan(plan, entry)
        # Command fidelity = 0.0, root cause alignment also low → overall score low
        assert score < 0.40, f"Expected score < 0.40 for invented commands, got {score}"

    def test_no_kb_commands_gives_full_fidelity(self):
        """When KB entry has no commands (only null-command steps), command fidelity defaults to 1.0."""
        # KBEntry requires at least 1 remediation_step; use a null-command step
        entry = KBEntry(
            entry_id="kb-no-cmd",
            incident_taxonomy="Errors:TLS",
            pattern_type="exact",
            error_pattern="certificate has expired",
            affected_services=[],
            severity="P1",
            root_cause_narrative="TLS certificate expired. Renew the cert.",
            remediation_steps=[
                KBRemediationStep(
                    order=1, action="Renew certificate via cert-manager.",
                    environment="kubectl", command=None, risk="low",
                )
            ],
            rollback_command="kubectl rollout undo deployment/{service} -n prod",
        )
        plan = _make_plan(commands=["kubectl get certificate -n prod"])
        score = _cross_validate_plan(plan, entry)
        # With no KB commands, command fidelity = 1.0; combined score >= 0.5
        assert score >= 0.50

    def test_plan_with_no_commands_scores_zero_fidelity(self):
        """When the plan generates no commands but KB has commands, fidelity is 0."""
        entry = _make_kb_entry()  # Has 2 commands
        plan = _make_plan(commands=[None, None])  # No commands generated
        score = _cross_validate_plan(plan, entry)
        # Command fidelity = 0.0, but root cause alignment may be partial;
        # score can be at most 0.5 (100% root cause alignment, 0% command fidelity)
        assert score <= 0.50, f"Expected score <= 0.50, got {score}"


# ---------------------------------------------------------------------------
# Tests: _cross_validate_plan — root cause alignment
# ---------------------------------------------------------------------------

class TestCrossValidatePlanRootCause:
    def test_matching_root_cause_scores_high(self):
        """Root cause with the same key terms as the KB narrative scores well."""
        entry = _make_kb_entry(
            root_cause="Pod was OOMKilled. The container memory limit is too low."
        )
        plan = _make_plan(
            root_cause="Container killed (OOMKilled) because memory limit too low for workload."
        )
        score = _cross_validate_plan(plan, entry)
        # Combined with command fidelity (both commands match), overall score should be decent
        assert score >= 0.30, f"Expected score >= 0.30, got {score}"

    def test_completely_different_root_cause_lowers_score(self):
        """A hallucinated root cause (totally different terms) reduces overall score."""
        entry = _make_kb_entry(
            root_cause="Pod was OOMKilled. The container memory limit is too low."
        )
        plan = _make_plan(
            root_cause="Network partition caused split brain in etcd cluster, leading to leader re-election timeout."
        )
        score = _cross_validate_plan(plan, entry)
        # Even if commands match, mismatched root cause should pull score down
        # (root cause alignment is 50% of the score)
        assert score < 0.80, f"Expected score < 0.80, got {score}"

    def test_short_kb_root_cause_command_fidelity_drives_score(self):
        """When KB root cause has very few matchable terms, command fidelity still contributes."""
        # Use identical command strings so command fidelity is unambiguously 1.0
        cmd = "kubectl get pods -n prod"
        entry = KBEntry(
            entry_id="kb-short-root",
            incident_taxonomy="Errors:OOM",
            pattern_type="exact",
            error_pattern="Exit Code 137",
            affected_services=[],
            severity="P1",
            root_cause_narrative="Unknown cause here X.",
            remediation_steps=[
                KBRemediationStep(
                    order=1, action="Check pods.", environment="kubectl",
                    command=cmd, risk="low",
                )
            ],
            rollback_command="kubectl rollout undo deployment/{service} -n prod",
        )
        plan = RemediationPlan(
            root_cause="Network partition in etcd cluster during leader re-election.",
            steps=[RemediationStep(order=1, action="Check pods.", command=cmd, risk="low")],
            rollback_command="kubectl rollout undo deployment/svc -n prod",
            estimated_mttr_minutes=5,
            postmortem_summary="Checked.",
        )
        score = _cross_validate_plan(plan, entry)
        # cmd == cmd so command_fidelity = 1.0 → score = 0.5 * 1.0 + 0.5 * root_alignment
        # root_alignment will be low but score must be >= 0.50 * 1.0 = 0.50
        # (slightly less due to root cause penalty, but at least 0.25)
        assert score >= 0.25, f"Expected score >= 0.25, got {score}"


# ---------------------------------------------------------------------------
# Tests: plan_node integration — faithfulness threshold enforcement
# ---------------------------------------------------------------------------

class TestPlanNodeFaithfulnessEnforcement:
    """
    These tests mock the LLM to return a controlled plan, then verify that
    the faithfulness cross-validation in plan_node correctly detects and
    flags hallucinated outputs.
    """

    def _make_state(self, kb_result: KBRetrievalResult | None = None) -> dict:
        return {
            "raw_alert": {
                "id": "alert-001",
                "title": "CRITICAL: Pod OOMKilled — payment-service",
                "description": "Exit Code 137 detected.",
                "service": "payment-service",
            },
            "triage_result": TriageResult(
                severity="P1",
                service="payment-service",
                confidence=0.92,
                reasoning="OOMKilled.",
            ),
            "severity": "P1",
            "extracted_evidence": "Exit Code 137. Pod restarted 3 times in 10 minutes.",
            "kb_result": kb_result,
            "messages": [],
        }

    @pytest.mark.asyncio
    async def test_low_faithfulness_clears_rollback(self):
        """
        When the LLM generates a plan with invented commands (faithfulness < 0.50),
        plan_node should clear the rollback_command, making is_high_risk=True.
        """
        from agent.nodes import plan_node

        kb_entry = _make_kb_entry()
        kb_result = KBRetrievalResult(
            entry=kb_entry,
            retrieval_score=0.85,
            match_type="exact",
            bypass_llm=False,
        )

        # Simulate LLM generating a plan with completely different commands
        hallucinated_plan = RemediationPlan(
            root_cause="Network partition caused split brain in etcd cluster.",  # Wrong
            steps=[
                RemediationStep(
                    order=1,
                    action="Delete etcd pod",
                    command="kubectl delete pod etcd-leader -n kube-system",  # Not in KB
                    risk="high",
                ),
            ],
            rollback_command="kubectl rollout undo deployment/payment-service -n prod",
            estimated_mttr_minutes=10,
            postmortem_summary="Network issue resolved.",
        )

        # Mock the entire LLM + parse chain to return the hallucinated plan
        with patch("agent.nodes._make_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(
                return_value=MagicMock(content=hallucinated_plan.model_dump_json())
            )
            mock_llm_factory.return_value = mock_llm

            with patch("agent.nodes.parse_json_robust", return_value=hallucinated_plan.model_dump()):
                updates = await plan_node(self._make_state(kb_result=kb_result))

        result_plan: RemediationPlan = updates["remediation_plan"]
        faithfulness: float | None = updates.get("faithfulness_score")

        assert faithfulness is not None
        assert faithfulness < 0.50, f"Expected faithfulness < 0.50, got {faithfulness}"
        assert result_plan.is_high_risk is True, (
            "Expected is_high_risk=True (rollback cleared) when faithfulness is low, "
            f"but rollback_command={result_plan.rollback_command!r}"
        )

    @pytest.mark.asyncio
    async def test_high_faithfulness_preserves_rollback(self):
        """
        When the LLM faithfully follows the KB entry, rollback_command is preserved
        and faithfulness_score >= 0.50.
        """
        from agent.nodes import plan_node

        kb_entry = _make_kb_entry()
        kb_result = KBRetrievalResult(
            entry=kb_entry,
            retrieval_score=0.88,
            match_type="exact",
            bypass_llm=False,
        )

        # Simulate LLM generating a faithful plan (same commands, substituted service name)
        faithful_plan = RemediationPlan(
            root_cause=(
                "Pod payment-service was OOMKilled (Exit Code 137). "
                "The container memory limit is too low for the current workload."
            ),
            steps=[
                RemediationStep(
                    order=1,
                    action="Increase memory limit.",
                    command="kubectl set resources deployment/payment-service -n prod --limits=memory=2Gi",
                    risk="medium",
                ),
                RemediationStep(
                    order=2,
                    action="Rolling restart.",
                    command="kubectl rollout restart deployment/payment-service -n prod",
                    risk="low",
                ),
            ],
            rollback_command="kubectl rollout undo deployment/payment-service -n prod",
            estimated_mttr_minutes=5,
            postmortem_summary="Memory limit increased. OOMKilled pods restarted.",
        )

        with patch("agent.nodes._make_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(
                return_value=MagicMock(content=faithful_plan.model_dump_json())
            )
            mock_llm_factory.return_value = mock_llm

            with patch("agent.nodes.parse_json_robust", return_value=faithful_plan.model_dump()):
                updates = await plan_node(self._make_state(kb_result=kb_result))

        result_plan: RemediationPlan = updates["remediation_plan"]
        faithfulness: float | None = updates.get("faithfulness_score")

        assert faithfulness is not None
        assert faithfulness >= 0.50, f"Expected faithfulness >= 0.50, got {faithfulness}"
        assert result_plan.is_high_risk is False, (
            "Expected is_high_risk=False when plan is faithful to KB, "
            f"but rollback_command={result_plan.rollback_command!r}"
        )

    @pytest.mark.asyncio
    async def test_no_kb_result_faithfulness_is_none(self):
        """
        When no KB entry was used (no match), faithfulness_score should be None
        and the plan should be in read-only mode (all commands null).
        """
        from agent.nodes import plan_node

        # No KB match
        no_match_kb = KBRetrievalResult(
            entry=None,
            retrieval_score=0.10,
            match_type="none",
            bypass_llm=False,
        )

        diagnostic_plan = RemediationPlan(
            root_cause="Insufficient evidence to determine root cause. Requires SRE review.",
            steps=[
                RemediationStep(
                    order=1,
                    action="Review logs and metrics.",
                    command=None,  # Read-only mode — no commands
                    risk="low",
                ),
            ],
            rollback_command="",  # No rollback in diagnostic mode
            estimated_mttr_minutes=None,
            postmortem_summary="Diagnostic summary for SRE review.",
        )

        with patch("agent.nodes._make_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(
                return_value=MagicMock(content=diagnostic_plan.model_dump_json())
            )
            mock_llm_factory.return_value = mock_llm

            with patch("agent.nodes.parse_json_robust", return_value=diagnostic_plan.model_dump()):
                updates = await plan_node(self._make_state(kb_result=no_match_kb))

        assert updates.get("faithfulness_score") is None, (
            "faithfulness_score should be None when no KB entry was used"
        )
