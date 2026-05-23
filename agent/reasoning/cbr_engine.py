"""
agent/reasoning/cbr_engine.py

Case-Based Reasoning (CBR) engine for the AIRS system.

Implements the four-phase CBR cycle:
  1. RETRIEVE  — Extract a feature vector from current telemetry and find the
                  most similar historical cases via cosine similarity.
  2. REUSE     — Adapt the best-matching historical remediation plan to the
                  current service/namespace/deployment context.
  3. REVISE    — (Handled by the logic_agent_node) Symbolic pruning of the
                  retrieved plan against current infrastructure invariants.
  4. RETAIN    — Store the resolved case back into the IncidentStore after
                  successful remediation (called from retain_node).

The CBR engine replaces pure LLM-generated remediation plans with
deterministic, human-validated solutions from historical data, ensuring
remediation safety and auditability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from agent.reasoning.incident_store import HistoricalCase, IncidentStore, VECTOR_DIM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature vector extraction constants
# ---------------------------------------------------------------------------

# Normalization clamps for each feature dimension
_VECTOR_CLAMPS = [
    100.0,    # [0] error_rate_pct
    100.0,    # [1] connection_pool_pct
    100.0,    # [2] cpu_utilization_pct
    100.0,    # [3] memory_utilization_pct
    10000.0,  # [4] dns_latency_ms
    100.0,    # [5] disk_utilization_pct
    1000.0,   # [6] ssl_error_count
    1000.0,   # [7] upstream_429_count
    1.0,      # [8] has_oom_signal (binary)
    1.0,      # [9] has_timeout_signal (binary)
    3.0,      # [10] service_tier
]

# Patterns for binary signal extraction
_OOM_PATTERNS = re.compile(
    r"(OOMKilled|OutOfMemoryError|heap\s+space|Exit\s+code\s+137|eviction|memory.*limit)",
    re.IGNORECASE,
)
_TIMEOUT_PATTERNS = re.compile(
    r"(TimeoutError|connection\s+timed\s+out|statement\s+timeout|pool\s+exhausted|QueuePool|timeout)",
    re.IGNORECASE,
)
_METRIC_PATTERNS = {
    "error_rate": re.compile(r"error.rate.*?(\d+\.?\d*)", re.IGNORECASE),
    "connection_pool": re.compile(r"pool.*?(\d+).*?%|connections.*?(\d+)", re.IGNORECASE),
    "cpu": re.compile(r"cpu.*?(\d+\.?\d*)\s*%", re.IGNORECASE),
    "memory": re.compile(r"memory.*?(\d+\.?\d*)\s*%", re.IGNORECASE),
    "dns_latency": re.compile(r"dns.*?(\d+)\s*ms|latency.*?(\d+)\s*ms", re.IGNORECASE),
    "disk": re.compile(r"disk.*?(\d+\.?\d*)\s*%|partition.*?(\d+)%", re.IGNORECASE),
    "ssl_errors": re.compile(r"ssl.*?(\d+)|certificate.*?(\d+)", re.IGNORECASE),
    "upstream_429": re.compile(r"429.*?(\d+)|rate.limit.*?(\d+)", re.IGNORECASE),
}


@dataclass
class ScoredCase:
    """A CBR retrieval result with its similarity score."""
    case: HistoricalCase
    similarity: float
    rank: int


@dataclass
class AdaptedPlan:
    """
    A remediation plan adapted from a historical case for the current incident context.
    """
    source_case_id: str
    source_service: str
    similarity_score: float
    root_cause_category: str
    adapted_steps: list[dict[str, Any]]
    adapted_rollback_command: str
    estimated_mttr_minutes: int
    postmortem_template: str
    confidence: float  # Overall CBR confidence (0.0-1.0)

    def to_markdown(self) -> str:
        """Render the adapted plan as markdown for LLM consumption."""
        lines = [
            f"## CBR-Adapted Remediation Plan",
            f"",
            f"**Precedent**: Incident `{self.source_case_id}` (service: `{self.source_service}`)",
            f"**Symptom Similarity**: {self.similarity_score:.0%}",
            f"**Root Cause Category**: `{self.root_cause_category}`",
            f"**CBR Confidence**: {self.confidence:.0%}",
            f"**Estimated MTTR**: {self.estimated_mttr_minutes} minutes",
            f"",
            f"### Adapted Steps",
        ]
        for step in self.adapted_steps:
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(step.get("risk", "low"), "⚪")
            lines.append(f"\n**Step {step['order']}**: {step['action']} {risk_icon}")
            if step.get("command"):
                lines.append(f"```bash\n{step['command']}\n```")
        lines += [
            f"",
            f"### Rollback Command",
            f"```bash\n{self.adapted_rollback_command}\n```",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CBREngine
# ---------------------------------------------------------------------------


class CBREngine:
    """
    Case-Based Reasoning engine for incident remediation.

    Rather than generating novel remediation plans from scratch (which risks
    unpredictable LLM outputs), the CBR engine matches the current incident
    against historical cases and adapts the proven solution to the current context.

    This enforces determinism and safety: every recommended action has been
    validated by a human engineer in a previous incident.
    """

    def __init__(self) -> None:
        self._store = IncidentStore.get_instance()

    # ------------------------------------------------------------------
    # Phase 1: RETRIEVE
    # ------------------------------------------------------------------

    def extract_feature_vector(
        self,
        telemetry: str,
        service: str,
        service_tier: int = 2,
    ) -> list[float]:
        """
        Extract a normalized feature vector from the current incident telemetry.

        The vector has VECTOR_DIM=11 dimensions (see incident_store.py for the
        index mapping). Values are normalized to [0.0, 1.0] by dividing by the
        corresponding clamp value in _VECTOR_CLAMPS.

        Args:
            telemetry: Concatenated telemetry string from investigate_node.
            service: Name of the affected service (used for tier lookup).
            service_tier: Tier of the service (1, 2, or 3).

        Returns:
            Normalized list[float] of length VECTOR_DIM.
        """
        raw = [0.0] * VECTOR_DIM

        # Extract numeric metrics via regex
        for m in _METRIC_PATTERNS["error_rate"].finditer(telemetry):
            val = float(next(g for g in m.groups() if g is not None))
            raw[0] = max(raw[0], min(val, 100.0))

        pool_match = _METRIC_PATTERNS["connection_pool"].search(telemetry)
        if pool_match:
            val = float(next(g for g in pool_match.groups() if g is not None))
            raw[1] = min(val, 100.0)

        cpu_match = _METRIC_PATTERNS["cpu"].search(telemetry)
        if cpu_match:
            val = float(next(g for g in cpu_match.groups() if g is not None))
            raw[2] = min(val, 100.0)

        mem_match = _METRIC_PATTERNS["memory"].search(telemetry)
        if mem_match:
            val = float(next(g for g in mem_match.groups() if g is not None))
            raw[3] = min(val, 100.0)

        dns_match = _METRIC_PATTERNS["dns_latency"].search(telemetry)
        if dns_match:
            val = float(next(g for g in dns_match.groups() if g is not None))
            raw[4] = min(val, 10000.0)

        disk_match = _METRIC_PATTERNS["disk"].search(telemetry)
        if disk_match:
            val = float(next(g for g in disk_match.groups() if g is not None))
            raw[5] = min(val, 100.0)

        ssl_match = _METRIC_PATTERNS["ssl_errors"].search(telemetry)
        if ssl_match:
            val = float(next(g for g in ssl_match.groups() if g is not None))
            raw[6] = min(val, 1000.0)

        rate_match = _METRIC_PATTERNS["upstream_429"].search(telemetry)
        if rate_match:
            val = float(next(g for g in rate_match.groups() if g is not None))
            raw[7] = min(val, 1000.0)

        # Binary signals
        raw[8] = 1.0 if _OOM_PATTERNS.search(telemetry) else 0.0
        raw[9] = 1.0 if _TIMEOUT_PATTERNS.search(telemetry) else 0.0

        # Service tier (normalized: tier1=1.0, tier2=0.67, tier3=0.33)
        raw[10] = float(service_tier)

        # Normalize each dimension by its clamp value
        normalized = [
            round(raw[i] / _VECTOR_CLAMPS[i], 4)
            for i in range(VECTOR_DIM)
        ]

        logger.debug(
            "[CBR] Feature vector for %s: oom=%s timeout=%s err_rate=%.1f pool=%.1f",
            service, raw[8], raw[9], raw[0], raw[1],
        )
        return normalized

    async def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[ScoredCase]:
        """
        Retrieve the most similar historical cases from the IncidentStore.

        Returns a ranked list of ScoredCase instances (highest similarity first).
        Returns empty list if no cases exceed the minimum similarity threshold.
        """
        raw_results = await self._store.search_similar(
            query_vector, top_k=top_k, min_similarity=min_similarity
        )
        scored = [
            ScoredCase(case=c, similarity=c.similarity_score, rank=i + 1)
            for i, c in enumerate(raw_results)
        ]
        if scored:
            logger.info(
                "[CBR] Retrieved %d case(s). Best match: %s (similarity=%.2f, category=%s)",
                len(scored),
                scored[0].case.incident_id,
                scored[0].similarity,
                scored[0].case.root_cause_category,
            )
        else:
            logger.info("[CBR] No similar cases found above threshold=%.2f", min_similarity)
        return scored

    # ------------------------------------------------------------------
    # Phase 2: REUSE / ADAPT
    # ------------------------------------------------------------------

    def reuse(
        self,
        current_service: str,
        current_namespace: str,
        best_match: HistoricalCase,
    ) -> AdaptedPlan:
        """
        Adapt the best-matching historical remediation plan to the current context.

        Substitutes:
          - Service name (e.g., 'payments-service' -> current_service)
          - Namespace (e.g., 'prod' -> current_namespace)
          - Keeps kubectl/shell command structure intact

        Args:
            current_service: Name of the service currently experiencing the incident.
            current_namespace: Kubernetes namespace of the current service.
            best_match: The historical case to adapt.

        Returns:
            AdaptedPlan with context-substituted steps and rollback command.
        """
        historical_service = best_match.service

        def _adapt_command(cmd: Optional[str]) -> Optional[str]:
            """Replace historical service/namespace references with current ones."""
            if not cmd:
                return cmd
            adapted = cmd
            adapted = adapted.replace(historical_service, current_service)
            adapted = adapted.replace("-n prod", f"-n {current_namespace}")
            adapted = adapted.replace("-n kube-system", "-n kube-system")  # system ns, don't change
            return adapted

        adapted_steps = []
        for step in best_match.remediation_steps:
            adapted_steps.append({
                **step,
                "command": _adapt_command(step.get("command")),
            })

        adapted_rollback = _adapt_command(best_match.rollback_command) or ""

        # Confidence is the similarity score, modulated by MTTR
        # (faster historical MTTR = higher confidence)
        confidence = round(
            best_match.similarity_score * min(1.0, 20.0 / max(1, best_match.mttr_minutes)),
            2
        )
        confidence = max(0.1, min(1.0, confidence))

        plan = AdaptedPlan(
            source_case_id=best_match.incident_id,
            source_service=historical_service,
            similarity_score=best_match.similarity_score,
            root_cause_category=best_match.root_cause_category,
            adapted_steps=adapted_steps,
            adapted_rollback_command=adapted_rollback,
            estimated_mttr_minutes=best_match.mttr_minutes,
            postmortem_template=best_match.postmortem_summary,
            confidence=confidence,
        )
        logger.info(
            "[CBR] Adapted plan from %s for %s (confidence=%.2f)",
            best_match.incident_id, current_service, confidence,
        )
        return plan

    # ------------------------------------------------------------------
    # Phase 4: RETAIN
    # ------------------------------------------------------------------

    async def retain(self, resolved_case: HistoricalCase) -> None:
        """
        Store a newly resolved case back into the IncidentStore.
        Called from retain_node after successful execution.
        """
        await self._store.store_case(resolved_case)
        logger.info("[CBR] Retained new case: %s", resolved_case.incident_id)

    # ------------------------------------------------------------------
    # Convenience: full retrieve-reuse pipeline
    # ------------------------------------------------------------------

    async def retrieve_and_adapt(
        self,
        telemetry: str,
        service: str,
        namespace: str = "prod",
        service_tier: int = 2,
    ) -> Optional[AdaptedPlan]:
        """
        Convenience method: extract feature vector, retrieve best match, and adapt.

        Returns None if no sufficiently similar historical case is found.
        """
        vector = self.extract_feature_vector(telemetry, service, service_tier)
        candidates = await self.retrieve(vector, top_k=1, min_similarity=0.5)
        if not candidates:
            return None
        return self.reuse(service, namespace, candidates[0].case)

    def format_candidates_markdown(self, candidates: list[ScoredCase]) -> str:
        """Render the ranked CBR candidates as a markdown summary for LLM context."""
        if not candidates:
            return "No similar historical cases found."
        lines = ["## CBR Candidate Matches\n"]
        for sc in candidates:
            lines.append(sc.case.summary_markdown())
            lines.append("")
        return "\n".join(lines)
