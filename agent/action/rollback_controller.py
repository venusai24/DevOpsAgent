"""
agent/action/rollback_controller.py

Automated rollback controller for the AIRS Action Layer.

Continuously monitors the four golden signals of observability during and
after remediation execution. If any signal degrades beyond its regression
threshold relative to the pre-remediation baseline, the controller
automatically executes the rollback command.

Four Golden Signals (as defined by Google SRE):
  1. Latency      — p99 response time
  2. Traffic      — requests per second
  3. Errors       — error rate percentage
  4. Saturation   — resource utilization percentage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agent.integrations.k8s import apply_kubectl_command
from agent.security.guardrails import is_safe_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression thresholds for each golden signal
# ---------------------------------------------------------------------------
# A regression is triggered when a signal's CURRENT value exceeds its
# BASELINE by more than the specified FACTOR (multiplicative threshold).

_REGRESSION_THRESHOLDS = {
    "latency_p99_ms": 1.5,     # 50% increase in latency triggers rollback
    "error_rate_pct": 2.0,     # 100% increase in error rate triggers rollback
    "saturation_pct": 1.2,     # 20% increase in saturation triggers rollback
}

# Absolute thresholds (trigger if value exceeds absolute limit regardless of baseline)
_ABSOLUTE_THRESHOLDS = {
    "error_rate_pct": 20.0,    # Always rollback if error rate > 20%
    "saturation_pct": 95.0,    # Always rollback if saturation > 95%
}


@dataclass
class GoldenSignalSnapshot:
    """A point-in-time snapshot of the four golden signals."""
    latency_p99_ms: float = 0.0
    traffic_rps: float = 0.0
    error_rate_pct: float = 0.0
    saturation_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "latency_p99_ms": self.latency_p99_ms,
            "traffic_rps": self.traffic_rps,
            "error_rate_pct": self.error_rate_pct,
            "saturation_pct": self.saturation_pct,
        }


@dataclass
class RollbackResult:
    """Result of a rollback controller health check."""
    should_rollback: bool
    reason: str
    regressed_signals: list[str]
    rollback_executed: bool = False
    rollback_output: str = ""

    def to_markdown(self) -> str:
        if not self.should_rollback:
            return "**✅ Golden signals healthy — no rollback required.**"
        status = "✅ Executed" if self.rollback_executed else "❌ Not executed"
        return (
            f"**🚨 ROLLBACK TRIGGERED** | {status}\n"
            f"- Reason: {self.reason}\n"
            f"- Regressed signals: {', '.join(self.regressed_signals)}"
        )


class RollbackController:
    """
    Monitors golden signals post-remediation and executes rollback if needed.

    In production, golden signals are fetched from Datadog/Prometheus.
    In demo mode, the controller simulates a healthy post-remediation state
    (since the mock API has static fixture data).
    """

    def check_regression(
        self,
        baseline: GoldenSignalSnapshot,
        current: GoldenSignalSnapshot,
    ) -> RollbackResult:
        """
        Compare current golden signals against the pre-remediation baseline.

        Args:
            baseline: Signals captured before remediation began.
            current: Signals captured after remediation was applied.

        Returns:
            RollbackResult indicating whether rollback is needed.
        """
        regressed: list[str] = []
        reasons: list[str] = []

        current_dict = current.to_dict()
        baseline_dict = baseline.to_dict()

        for signal, threshold_factor in _REGRESSION_THRESHOLDS.items():
            base_val = baseline_dict.get(signal, 0.0)
            curr_val = current_dict.get(signal, 0.0)

            # Check multiplicative regression
            if base_val > 0 and curr_val > base_val * threshold_factor:
                regressed.append(signal)
                reasons.append(
                    f"{signal} increased {curr_val/base_val:.1f}x (threshold: {threshold_factor}x)"
                )

            # Check absolute threshold
            abs_limit = _ABSOLUTE_THRESHOLDS.get(signal)
            if abs_limit and curr_val > abs_limit:
                if signal not in regressed:
                    regressed.append(signal)
                reasons.append(f"{signal}={curr_val:.1f} exceeds absolute limit={abs_limit}")

        should_rollback = bool(regressed)
        return RollbackResult(
            should_rollback=should_rollback,
            reason="; ".join(reasons) if reasons else "All golden signals within acceptable bounds.",
            regressed_signals=regressed,
        )

    async def monitor_and_rollback(
        self,
        baseline: GoldenSignalSnapshot,
        current: GoldenSignalSnapshot,
        rollback_command: str,
    ) -> RollbackResult:
        """
        Full pipeline: check regression, then execute rollback if triggered.

        Args:
            baseline: Pre-remediation golden signal snapshot.
            current: Post-remediation golden signal snapshot.
            rollback_command: The kubectl/shell command to execute on regression.

        Returns:
            RollbackResult with execution status.
        """
        result = self.check_regression(baseline, current)

        if result.should_rollback:
            logger.warning(
                "[RollbackController] Regression detected in signals=%s. Executing rollback.",
                result.regressed_signals,
            )
            if rollback_command and is_safe_command(rollback_command):
                try:
                    output = apply_kubectl_command(rollback_command)
                    result.rollback_executed = True
                    result.rollback_output = str(output)
                    logger.info("[RollbackController] Rollback executed: %s", output)
                except Exception as exc:
                    result.rollback_output = f"[ROLLBACK FAILED] {exc}"
                    logger.error("[RollbackController] Rollback failed: %s", exc)
            else:
                logger.warning("[RollbackController] Rollback command blocked by guardrail: %r", rollback_command[:80])

        return result

    @staticmethod
    def capture_demo_baseline(service: str) -> GoldenSignalSnapshot:
        """
        Return a simulated pre-incident baseline for demo mode.
        In production, this would query Datadog/Prometheus for the 30-minute p50.
        """
        return GoldenSignalSnapshot(
            latency_p99_ms=120.0,
            traffic_rps=1500.0,
            error_rate_pct=0.3,
            saturation_pct=35.0,
        )

    @staticmethod
    def capture_demo_post_remediation(remediation_succeeded: bool = True) -> GoldenSignalSnapshot:
        """
        Return simulated post-remediation signals for demo mode.
        Healthy state if remediation_succeeded=True, regressed state otherwise.
        """
        if remediation_succeeded:
            return GoldenSignalSnapshot(
                latency_p99_ms=115.0,
                traffic_rps=1520.0,
                error_rate_pct=0.2,
                saturation_pct=30.0,
            )
        return GoldenSignalSnapshot(
            latency_p99_ms=980.0,
            traffic_rps=800.0,
            error_rate_pct=45.0,
            saturation_pct=98.0,
        )
