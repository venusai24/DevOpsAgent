"""
airs_v2/action/health_monitor.py
==================================

HealthMonitor — Post-Execution Health Check Oracle
----------------------------------------------------

Runs a set of service health probes after a plan's approved actions have been
dispatched.  If **any** probe returns ``healthy=False``, the
``ExecutionEngine`` triggers a mandatory rollback.

Production vs. test mode
-------------------------
In production, the monitor would call real liveness/readiness endpoints
(e.g. Kubernetes readiness probes via the k8s API, or HTTP ``/health``
endpoints via ``httpx``).

In tests, pass ``_inject_results`` — a ``dict[service_name, bool]`` — to
override the probe outcomes.  This makes health-check-triggered rollback tests
fully deterministic without any real network calls.

::

    monitor = HealthMonitor()
    results = await monitor.run_checks(
        ["payments-service", "api-gateway"],
        _inject_results={"payments-service": False, "api-gateway": True},
    )
    assert monitor.any_unhealthy(results)   # True — payments-service is down

Design invariants
-----------------
* ``run_checks`` is always async (consistent with the production HTTP shape).
* All services in ``services`` are probed; none are skipped.
* An injected ``False`` produces ``HealthCheckResult(healthy=False, http_status=503)``.
* If ``_inject_results`` is omitted, ALL services are assumed healthy
  (stub behaviour for unit tests that don't care about rollback).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from airs_v2.action.types import HealthCheckResult

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Post-execution health oracle.

    Parameters
    ----------
    probe_timeout_s:
        Maximum seconds to wait for a single probe response.  Used only in
        production mode (no effect when ``_inject_results`` is provided).
    """

    def __init__(self, probe_timeout_s: float = 5.0) -> None:
        self._timeout = probe_timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_checks(
        self,
        services: list[str],
        *,
        _inject_results: dict[str, bool] | None = None,
    ) -> list[HealthCheckResult]:
        """
        Run health probes for each service in *services*.

        Parameters
        ----------
        services:
            List of service names to probe.
        _inject_results:
            **Test-only override.**  Maps ``service_name → healthy (bool)``.
            Services not in the dict default to ``healthy=True``.  Pass an
            explicit ``False`` to simulate an unhealthy service.

        Returns
        -------
        list[HealthCheckResult]
            One result per service, in the same order as *services*.
        """
        results: list[HealthCheckResult] = []

        for svc in services:
            result = await self._probe(svc, inject=_inject_results)
            results.append(result)
            status_icon = "✅" if result.healthy else "❌"
            logger.info(
                "[HealthMonitor] %s service=%s http=%d latency=%.1f ms",
                status_icon,
                svc,
                result.http_status,
                result.latency_ms,
            )

        unhealthy_count = sum(1 for r in results if not r.healthy)
        logger.info(
            "[HealthMonitor] probe complete: total=%d healthy=%d unhealthy=%d",
            len(results),
            len(results) - unhealthy_count,
            unhealthy_count,
        )
        return results

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def any_unhealthy(results: list[HealthCheckResult]) -> bool:
        """Return True if any result in *results* is unhealthy."""
        return any(not r.healthy for r in results)

    @staticmethod
    def unhealthy_services(results: list[HealthCheckResult]) -> list[str]:
        """Return the names of all unhealthy services."""
        return [r.service for r in results if not r.healthy]

    # ------------------------------------------------------------------
    # Internal — probe implementation
    # ------------------------------------------------------------------

    async def _probe(
        self,
        service: str,
        *,
        inject: dict[str, bool] | None,
    ) -> HealthCheckResult:
        """
        Probe a single service.

        When ``inject`` is provided, use the injected value.
        Otherwise, assume healthy (stub for unit tests).
        In a real implementation this would call the k8s readiness endpoint.
        """
        checked_at = datetime.now(timezone.utc).isoformat()

        if inject is not None:
            healthy = inject.get(service, True)  # services not listed default healthy
            if healthy:
                return HealthCheckResult(
                    service=service,
                    healthy=True,
                    http_status=200,
                    latency_ms=1.0,
                    checked_at=checked_at,
                )
            else:
                return HealthCheckResult(
                    service=service,
                    healthy=False,
                    http_status=503,
                    latency_ms=0.0,
                    error_message=(
                        f"Service '{service}' failed health probe "
                        f"(injected failure for deterministic test)."
                    ),
                    checked_at=checked_at,
                )

        # Production stub: assume healthy (real implementation would use httpx)
        # In production: response = await httpx.get(f"http://{service}/readyz", timeout=self._timeout)
        logger.debug(
            "[HealthMonitor] No inject map — assuming %s is healthy (stub mode)", service
        )
        return HealthCheckResult(
            service=service,
            healthy=True,
            http_status=200,
            latency_ms=2.0,
            checked_at=checked_at,
        )
