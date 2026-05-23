"""
agent/action/canary_controller.py

Progressive canary deployment controller for the AIRS Action Layer.

Rather than applying remediation changes atomically to 100% of traffic,
the canary controller progressively shifts traffic in stages, monitoring
the four golden signals at each stage. If any signal regresses beyond
its threshold, the canary is immediately halted and rolled back.

Canary Stages (configurable):
  Stage 1: 5% of traffic  — observe for 30 seconds
  Stage 2: 25% of traffic — observe for 60 seconds
  Stage 3: 50% of traffic — observe for 60 seconds
  Stage 4: 100% of traffic — final confirmation

In demo mode, the controller uses mock golden signal data from the fixture
database to simulate canary behaviour without requiring live Kubernetes infra.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from agent.action.rollback_controller import (
    GoldenSignalSnapshot,
    RollbackController,
    RollbackResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canary stage definitions
# ---------------------------------------------------------------------------

CANARY_STAGES: list[dict[str, Any]] = [
    {"stage": 1, "traffic_pct": 5,   "observe_seconds": 30,  "label": "5%  canary"},
    {"stage": 2, "traffic_pct": 25,  "observe_seconds": 60,  "label": "25% canary"},
    {"stage": 3, "traffic_pct": 50,  "observe_seconds": 60,  "label": "50% canary"},
    {"stage": 4, "traffic_pct": 100, "observe_seconds": 30,  "label": "100% full rollout"},
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Outcome of a single canary stage."""
    stage: int
    traffic_pct: int
    signals: GoldenSignalSnapshot
    healthy: bool
    observations: str  # Human-readable summary of what was observed


@dataclass
class CanaryResult:
    """Aggregated result of the full canary deployment process."""
    service: str
    succeeded: bool
    stage_results: list[StageResult] = field(default_factory=list)
    halted_at_stage: Optional[int] = None
    rollback_result: Optional[RollbackResult] = None
    total_duration_seconds: float = 0.0
    final_signal_health: Optional[GoldenSignalSnapshot] = None

    def to_markdown(self) -> str:
        """Render the canary result as a markdown report."""
        status = "✅ Canary succeeded \u2014 100% rollout complete" if self.succeeded else \
                 f"🚨 Canary halted at stage {self.halted_at_stage} \u2014 automatic rollback triggered"
        lines = [
            f"## Canary Deployment: `{self.service}`",
            f"**Status**: {status}",
            f"**Duration**: {self.total_duration_seconds:.0f}s",
            "",
            "### Stage Results",
        ]
        for sr in self.stage_results:
            icon = "✅" if sr.healthy else "❌"
            lines.append(
                f"{icon} **Stage {sr.stage}** ({sr.traffic_pct}% traffic): "
                f"err={sr.signals.error_rate_pct:.1f}% lat={sr.signals.latency_p99_ms:.0f}ms "
                f"sat={sr.signals.saturation_pct:.0f}%"
            )
            lines.append(f"   ↳ {sr.observations}")
        if self.rollback_result:
            lines.append(f"\n{self.rollback_result.to_markdown()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CanaryController
# ---------------------------------------------------------------------------


class CanaryController:
    """
    Progressive canary deployment controller.

    Applies a remediation command in stages, monitoring golden signals at
    each stage and rolling back automatically on regression.

    In production mode, golden signals would be fetched from Datadog or
    Prometheus via the existing integrations. In demo mode, the controller
    uses simulated signals that reflect the expected post-remediation state.

    Usage:
        ctrl = CanaryController()
        result = await ctrl.execute_canary(
            service="payments-service",
            remediation_command="kubectl rollout restart deployment/payments-service -n prod",
            rollback_command="kubectl rollout undo deployment/payments-service -n prod",
            baseline=GoldenSignalSnapshot(latency_p99_ms=120.0, error_rate_pct=0.3, ...),
        )
    """

    def __init__(self, demo_mode: bool = True) -> None:
        """
        Args:
            demo_mode: If True, use simulated golden signals instead of live metrics.
                       Set to False in production when Datadog/Prometheus is available.
        """
        self._demo_mode = demo_mode
        self._rollback_ctrl = RollbackController()

    async def execute_canary(
        self,
        service: str,
        remediation_command: str,
        rollback_command: str,
        baseline: Optional[GoldenSignalSnapshot] = None,
        stages: Optional[list[dict]] = None,
    ) -> CanaryResult:
        """
        Execute a progressive canary deployment.

        Args:
            service: The service being remediated.
            remediation_command: The command to execute at each stage.
            rollback_command: The rollback command to run if regression is detected.
            baseline: Pre-incident golden signal snapshot. If None, uses demo baseline.
            stages: Custom stage definitions. Defaults to CANARY_STAGES.

        Returns:
            CanaryResult with per-stage observations and final status.
        """
        start_time = datetime.utcnow()
        stages = stages or CANARY_STAGES
        baseline = baseline or RollbackController.capture_demo_baseline(service)

        logger.info(
            "[Canary] Starting canary for %s: %d stages, command=%r",
            service, len(stages), remediation_command[:80],
        )

        stage_results: list[StageResult] = []
        halted_at: Optional[int] = None
        rollback_result: Optional[RollbackResult] = None

        for stage_def in stages:
            stage_num = stage_def["stage"]
            traffic_pct = stage_def["traffic_pct"]
            observe_secs = stage_def["observe_seconds"]
            label = stage_def["label"]

            logger.info("[Canary] Stage %d: shifting %d%% traffic — observing for %ds",
                        stage_num, traffic_pct, observe_secs)

            # In production: apply traffic weight to Istio VirtualService or NGINX weights
            # In demo: simulate the traffic shift
            if not self._demo_mode:
                await self._apply_traffic_shift(service, traffic_pct, remediation_command)

            # Observe golden signals
            # In demo mode: simulate improving signals as traffic increases
            if self._demo_mode:
                # Simulate a brief delay to make the canary feel real
                await asyncio.sleep(min(2, observe_secs / 10))
                signals = self._simulate_canary_signals(baseline, traffic_pct)
            else:
                await asyncio.sleep(observe_secs)
                signals = await self._fetch_live_signals(service)

            # Check for regression
            regression_check = self._rollback_ctrl.check_regression(baseline, signals)

            observations = self._build_stage_summary(stage_num, traffic_pct, signals, regression_check)

            stage_result = StageResult(
                stage=stage_num,
                traffic_pct=traffic_pct,
                signals=signals,
                healthy=not regression_check.should_rollback,
                observations=observations,
            )
            stage_results.append(stage_result)

            if regression_check.should_rollback:
                logger.warning(
                    "[Canary] Stage %d regression detected: %s. Triggering rollback.",
                    stage_num, regression_check.reason,
                )
                halted_at = stage_num
                # Execute rollback
                rollback_result = await self._rollback_ctrl.monitor_and_rollback(
                    baseline, signals, rollback_command
                )
                break

            logger.info("[Canary] Stage %d healthy. Advancing.", stage_num)

        duration = (datetime.utcnow() - start_time).total_seconds()
        succeeded = halted_at is None

        result = CanaryResult(
            service=service,
            succeeded=succeeded,
            stage_results=stage_results,
            halted_at_stage=halted_at,
            rollback_result=rollback_result,
            total_duration_seconds=duration,
            final_signal_health=stage_results[-1].signals if stage_results else None,
        )

        logger.info(
            "[Canary] Completed: service=%s succeeded=%s duration=%.0fs stages_completed=%d/%d",
            service, succeeded, duration, len(stage_results), len(stages),
        )
        return result

    async def _apply_traffic_shift(
        self, service: str, traffic_pct: int, command: str
    ) -> None:
        """Apply a traffic weight shift (production implementation placeholder)."""
        # Production: update Istio VirtualService weights or NGINX upstream weights
        logger.info("[Canary] Applying traffic shift: %s -> %d%%", service, traffic_pct)

    async def _fetch_live_signals(self, service: str) -> GoldenSignalSnapshot:
        """Fetch live golden signals from Datadog/Prometheus (production stub)."""
        # Production: query Datadog API or Prometheus /query endpoint
        logger.info("[Canary] Fetching live signals for %s", service)
        return RollbackController.capture_demo_post_remediation(True)

    def _simulate_canary_signals(
        self, baseline: GoldenSignalSnapshot, traffic_pct: int
    ) -> GoldenSignalSnapshot:
        """
        Simulate progressive signal improvement as canary traffic increases.

        Models the expected behavior: error rate drops as more traffic uses
        the fixed version, latency normalizes, saturation decreases.
        """
        # Simulate improvement proportional to traffic shifted to new version
        improvement_factor = traffic_pct / 100.0
        return GoldenSignalSnapshot(
            latency_p99_ms=max(50.0, baseline.latency_p99_ms * (1 - 0.3 * improvement_factor)),
            traffic_rps=baseline.traffic_rps * (0.9 + 0.1 * improvement_factor),
            error_rate_pct=max(0.1, 47.0 * (1 - improvement_factor) + 0.2 * improvement_factor),
            saturation_pct=max(15.0, 85.0 * (1 - improvement_factor) + 15.0 * improvement_factor),
        )

    def _build_stage_summary(
        self,
        stage: int,
        traffic_pct: int,
        signals: GoldenSignalSnapshot,
        regression: RollbackResult,
    ) -> str:
        """Build a one-line human-readable stage summary."""
        if regression.should_rollback:
            return f"REGRESSION: {regression.reason}"
        health_indicators = []
        if signals.error_rate_pct < 1.0:
            health_indicators.append("errors nominal")
        if signals.latency_p99_ms < 500:
            health_indicators.append("latency healthy")
        if signals.saturation_pct < 80:
            health_indicators.append("saturation OK")
        return f"All signals healthy — {', '.join(health_indicators)}" if health_indicators else "Signals within bounds"
