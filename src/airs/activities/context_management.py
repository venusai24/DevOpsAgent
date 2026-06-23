"""
Context Management Activity — Module 1.10.

Temporal activity: Admit a new EvidenceCandidate into the investigation context
and run eviction if needed. This is the only place where context mutations happen.

Temporal contract:
  - Deterministic: same candidate + state always produces same output
  - Pure state transition: receives InvestigationState, returns updated state
  - No external I/O (entity resolution uses Union-Find, Redis as cache)
"""
from __future__ import annotations

import logging

from temporalio import activity

from airs.context.manager import ProactiveContextManager
from airs.entity_resolution.engine import get_entity_resolution_engine
from airs.models.context import ContextMetrics
from airs.models.evidence import EvidenceCandidate
from airs.models.investigation import InvestigationState

log = logging.getLogger(__name__)


@activity.defn(name="admit_evidence_candidate")
async def admit_evidence_candidate(
    state: InvestigationState,
    candidate: EvidenceCandidate,
) -> tuple[InvestigationState, ContextMetrics]:
    """
    Admit a pre-filtered EvidenceCandidate into the active evidence tier.

    Steps:
    1. Entity resolution: map raw identifiers → canonical GUID
    2. Budget check: verify token headroom in active_evidence partition
    3. Build EvidenceNode from candidate
    4. Score node via ContextScorer
    5. Add to InsightTiers.active + InvestigationGraph
    6. Run eviction if context is under pressure

    Args:
        state:     Current InvestigationState.
        candidate: Pre-filtered EvidenceCandidate from execute_mcp_tool activity.

    Returns:
        Tuple of (updated InvestigationState, ContextMetrics snapshot).
    """
    activity.logger.info(
        "Admitting evidence candidate from %s/%s (hop=%d, tokens=%d)",
        candidate.mcp_server_id, candidate.tool_invoked,
        candidate.hop_index, candidate.token_count,
    )

    # Get singletons
    er_engine = get_entity_resolution_engine()
    ctx_manager = ProactiveContextManager.from_settings()

    # Admit candidate
    new_state = ctx_manager.admit_candidate(
        state=state,
        candidate=candidate,
        entity_resolution_engine=er_engine,
    )

    # Evaluate pressure and evict if needed
    new_state, metrics = ctx_manager.evaluate_and_evict(new_state)

    activity.logger.info(
        "Context post-admission: active=%d, summarized=%d, "
        "utilization=%.1f%%, evictions=%d",
        len(new_state.insight_tiers.active),
        len(new_state.insight_tiers.summarized),
        metrics.utilization_ratio * 100,
        metrics.eviction_count,
    )

    return new_state, metrics
