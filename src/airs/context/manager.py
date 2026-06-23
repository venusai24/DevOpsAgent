"""
Context Manager — Module 1.7.

The ProactiveContextManager is the central decision authority for context
window management during the investigation lifecycle.

Responsibilities:
  1. ADMIT — evaluate if a new EvidenceCandidate can enter the active tier
  2. EVICT  — pressure-triggered eviction of low-scoring evidence to summarized/archived
  3. SCORE  — compute composite relevance scores for all active evidence
  4. PRUNE  — evict the lowest-scoring non-causal evidence when under pressure
  5. TELESCOPE — compress middle causal chain nodes when causal budget overflows

All operations return a NEW InvestigationState (immutable update pattern).
No in-place mutation — the Temporal workflow serializes state after each call.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from airs.context.budget import BudgetTracker
from airs.context.scorer import ContextScorer
from airs.models.context import (
    ContextBudget,
    ContextMetrics,
    InsightSummary,
    InsightTier,
    InsightTiers,
    TelescopedSummaryNode,
)
from airs.models.evidence import (
    EntityDescriptor,
    EntityType,
    EvidenceCandidate,
    EvidenceNode,
    ProvenanceRecord,
    SignalSource,
    TemporalityBound,
    UncertaintyMetrics,
)
from airs.models.investigation import InvestigationState

log = logging.getLogger(__name__)


class ProactiveContextManager:
    """
    Manages the 4-tier context memory (Active, Summarized, Archived, Discarded)
    and enforces the token budget across all partitions.

    Usage in Temporal activities:
        manager = ProactiveContextManager.from_settings()
        new_state = manager.admit_candidate(state, candidate, er_engine)
        new_state = manager.evaluate_and_evict(new_state)
    """

    def __init__(
        self,
        scorer: ContextScorer,
        pressure_threshold: float = 0.85,
        eviction_score_threshold: float = 0.30,
    ) -> None:
        self._scorer = scorer
        self._pressure_threshold = pressure_threshold
        self._eviction_threshold = eviction_score_threshold

    @classmethod
    def from_settings(cls) -> "ProactiveContextManager":
        """Create ProactiveContextManager from calibration defaults."""
        from airs.risk.calibration import get_calibration_store
        store = get_calibration_store()
        scorer = ContextScorer.from_calibration()
        return cls(
            scorer=scorer,
            pressure_threshold=store.get_eviction_pressure_threshold(),
            eviction_score_threshold=store.get_eviction_score_threshold(),
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def admit_candidate(
        self,
        state: InvestigationState,
        candidate: EvidenceCandidate,
        entity_resolution_engine: Any,
    ) -> InvestigationState:
        """
        Attempt to admit an EvidenceCandidate into the active evidence tier.

        Steps:
        1. Resolve entity to canonical GUID via ER engine
        2. Build tracker from current state
        3. Check token headroom in active evidence partition
        4. If headroom available: promote to EvidenceNode, add to active tier
        5. If no headroom: discard (do NOT evict — eviction is separate)

        Args:
            state:                    Current investigation state.
            candidate:                Pre-filtered candidate from SignalAdapter.
            entity_resolution_engine: EntityResolutionEngine for GUID resolution.

        Returns:
            New InvestigationState with (potentially) the node admitted.
        """
        tracker = BudgetTracker(state.context_budget)
        self._sync_tracker(tracker, state)

        # Check token headroom
        if not tracker.can_admit("active_evidence", candidate.token_count):
            log.warning(
                "Context budget exhausted — discarding candidate from %s/%s",
                candidate.mcp_server_id,
                candidate.tool_invoked,
            )
            # Increment discard count
            new_tiers = state.insight_tiers.model_copy(
                update={"discarded_count": state.insight_tiers.discarded_count + 1}
            )
            return state.model_copy(update={"insight_tiers": new_tiers})

        # Resolve entity
        try:
            canonical_guid = entity_resolution_engine.resolve_entity(
                candidate.entity_type,
                candidate.raw_identifiers,
            )
        except Exception as exc:
            log.warning("Entity resolution failed: %s — using random GUID", exc)
            canonical_guid = uuid.uuid4()

        # Build EvidenceNode
        node = self._build_evidence_node(candidate, canonical_guid)

        # Compute context score and store it on the node
        score = self._scorer.score_node(node, state.graph, state.total_hop_count)
        node = node.model_copy(update={"context_score": score.composite})

        # Add to active tier
        new_active = list(state.insight_tiers.active) + [node]
        new_tiers = state.insight_tiers.model_copy(update={"active": new_active})

        # Add to investigation graph
        new_graph = state.graph.model_copy(deep=True)
        new_graph.add_node(node)

        log.debug(
            "Admitted evidence node %s (score=%.3f, tokens=%d)",
            node.node_id,
            score.composite,
            candidate.token_count,
        )

        return state.model_copy(
            update={
                "insight_tiers": new_tiers,
                "graph": new_graph,
                "last_updated_at": datetime.now(timezone.utc),
            }
        )

    def evaluate_and_evict(
        self,
        state: InvestigationState,
    ) -> tuple[InvestigationState, ContextMetrics]:
        """
        Evaluate context pressure and evict if necessary.

        Called after each evidence admission. If the active evidence tier
        is above the pressure threshold, evict the lowest-scoring non-causal
        nodes to the summarized or archived tier.

        Returns:
            (new_state, context_metrics) — updated state and current metrics.
        """
        tracker = BudgetTracker(state.context_budget)
        self._sync_tracker(tracker, state)

        active_nodes: list[EvidenceNode] = list(state.insight_tiers.active)
        if not active_nodes:
            metrics = tracker.compute_metrics()
            return state, metrics

        # Compute scores for all active nodes
        scores = self._scorer.score_all_nodes(
            active_nodes, state.graph, state.total_hop_count
        )

        # If not under pressure, just compute metrics and return
        if not tracker.is_under_pressure(self._pressure_threshold):
            avg_score = (
                sum(s.composite for s in scores.values()) / len(scores)
                if scores else 0.0
            )
            metrics = tracker.compute_metrics(avg_composite_score=avg_score)
            return state, metrics

        log.info(
            "Context pressure triggered eviction: active_evidence utilisation=%.1f%%",
            tracker.get_utilisation("active_evidence") * 100,
        )

        # Sort by composite score (ascending — lowest first for eviction)
        ranked = sorted(
            active_nodes,
            key=lambda n: scores.get(n.node_id, type("", (), {"composite": 0.0})()).composite
            if n.node_id in scores else 0.0,
        )

        new_active = list(active_nodes)
        new_summarized = list(state.insight_tiers.summarized)
        new_archived = list(state.insight_tiers.archived)
        eviction_count = 0

        for node in ranked:
            if not tracker.is_under_pressure(self._pressure_threshold):
                break  # Pressure relieved

            score = scores.get(node.node_id)
            if score is None:
                continue

            # Never evict causal chain nodes
            if node.is_causal or node.node_id in state.graph.causal_chain_ids:
                continue

            if score.composite < self._eviction_threshold:
                # Evict to archived (too low quality even for summarized)
                new_active.remove(node)
                new_archived.append(node.node_id)
                tracker.subtract_from_partition(
                    "active_evidence",
                    node.context.get("_token_count", 100),
                )
                eviction_count += 1
                log.debug("Archived node %s (score=%.3f)", node.node_id, score.composite)
            else:
                # Evict to summarized tier
                new_active.remove(node)
                summary = InsightSummary(
                    original_node_ids=[node.node_id],
                    hop_range=(node.hop_index, node.hop_index),
                    summary_text=self._make_summary_text(node),
                    key_services=[i for i in node.entity.observed_identifiers[:2]],
                    key_evidence_types=[node.signal_source.value],
                    aggregate_confidence_delta=score.composite,
                    token_count=min(50, node.context.get("_token_count", 100) // 4),
                )
                new_summarized.append(summary)
                tracker.subtract_from_partition(
                    "active_evidence",
                    node.context.get("_token_count", 100),
                )
                tracker.add_to_partition("summarized_evidence", summary.token_count)
                eviction_count += 1
                log.debug("Summarized node %s (score=%.3f)", node.node_id, score.composite)

        new_tiers = state.insight_tiers.model_copy(
            update={
                "active": new_active,
                "summarized": new_summarized,
                "archived": new_archived,
            }
        )

        avg_score = (
            sum(scores[n.node_id].composite for n in new_active if n.node_id in scores)
            / len(new_active)
            if new_active else 0.0
        )
        metrics = tracker.compute_metrics(avg_composite_score=avg_score)
        metrics = metrics.model_copy(update={"eviction_count": eviction_count})

        return (
            state.model_copy(
                update={
                    "insight_tiers": new_tiers,
                    "last_updated_at": datetime.now(timezone.utc),
                }
            ),
            metrics,
        )

    def get_context_window(self, state: InvestigationState) -> dict:
        """
        Assemble the current context window content for LLM consumption.

        Returns structured dict with all 4 tier contents, ordered by priority.
        The output is directly injected into the LangGraph prompt.
        """
        active_nodes = state.insight_tiers.active
        summaries = state.insight_tiers.summarized
        archived_ids = state.insight_tiers.archived
        causal_chain = state.graph.get_causal_chain()

        # Sort active nodes: causal chain first, then by context score descending
        causal_ids = set(state.graph.causal_chain_ids)
        causal_active = [n for n in active_nodes if n.node_id in causal_ids]
        non_causal_active = sorted(
            [n for n in active_nodes if n.node_id not in causal_ids],
            key=lambda n: n.context_score or 0.0,
            reverse=True,
        )

        return {
            "causal_chain": [self._node_to_context_dict(n) for n in causal_active],
            "active_evidence": [self._node_to_context_dict(n) for n in non_causal_active],
            "summarized_evidence": [
                {
                    "hop_range": s.hop_range,
                    "summary": s.summary_text,
                    "key_services": s.key_services,
                    "confidence_delta": s.aggregate_confidence_delta,
                }
                for s in summaries
            ],
            "archived_node_ids": archived_ids,
            "total_active": len(active_nodes),
            "total_summarized": len(summaries),
            "total_archived": len(archived_ids),
            "discarded": state.insight_tiers.discarded_count,
        }

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _sync_tracker(
        self, tracker: BudgetTracker, state: InvestigationState
    ) -> None:
        """Sync the budget tracker with actual token usage from state."""
        active_tokens = sum(
            (n.context.get("_token_count", 100) for n in state.insight_tiers.active),
            0,
        )
        summarized_tokens = sum(
            (s.token_count for s in state.insight_tiers.summarized), 0
        )
        tracker.set_partition_usage("active_evidence", active_tokens)
        tracker.set_partition_usage("summarized_evidence", summarized_tokens)

    @staticmethod
    def _build_evidence_node(
        candidate: EvidenceCandidate,
        canonical_guid: uuid.UUID,
    ) -> EvidenceNode:
        """Convert an EvidenceCandidate to a full EvidenceNode."""
        from airs.models.evidence import EntityDescriptor, ProvenanceRecord, TemporalityBound, UncertaintyMetrics

        node_id = hashlib.sha256(
            f"{candidate.idempotency_key}:{candidate.hop_index}".encode()
        ).hexdigest()[:24]

        return EvidenceNode(
            node_id=node_id,
            entity=EntityDescriptor(
                canonical_guid=canonical_guid,
                entity_type=candidate.entity_type,
                observed_identifiers=candidate.raw_identifiers,
            ),
            signal_source=candidate.signal_source,
            context={
                **candidate.filtered_content,
                "_token_count": candidate.token_count,
            },
            uncertainty=UncertaintyMetrics(
                nonconformity_score=0.50,   # Will be updated after LLM interpretation
                calibration_quantile=0.50,
                li_score=0.50,
                bert_f1_score=0.50,
            ),
            provenance=ProvenanceRecord(
                mcp_server_id=candidate.mcp_server_id,
                tool_invoked=candidate.tool_invoked,
                query_parameters_hash=candidate.query_parameters_hash,
                raw_response_digest=candidate.raw_response_digest,
                idempotency_key=candidate.idempotency_key,
            ),
            temporality=TemporalityBound(
                observed_at=candidate.observed_at,
                valid_from=candidate.valid_from,
                valid_until=candidate.valid_until,
            ),
            hop_index=candidate.hop_index,
            is_causal=False,
        )

    @staticmethod
    def _make_summary_text(node: EvidenceNode) -> str:
        """Generate a brief summary text for a node being moved to summarized tier."""
        entity_ids = node.entity.observed_identifiers[:2]
        entity_str = ", ".join(entity_ids) if entity_ids else str(node.entity.canonical_guid)
        signal = node.signal_source.value.lower()
        hop = node.hop_index

        # Extract key finding from context
        content = node.context
        metric = content.get("metric_name", content.get("metric_name", ""))
        error_info = content.get("log_patterns", content.get("error_spans", ""))

        summary = f"[Hop {hop}] {signal} evidence for {entity_str}"
        if metric:
            summary += f": {metric}"
        return summary

    @staticmethod
    def _node_to_context_dict(node: EvidenceNode) -> dict:
        """Convert an EvidenceNode to a slim LLM-consumable dict."""
        return {
            "node_id": node.node_id,
            "hop": node.hop_index,
            "entity": ", ".join(node.entity.observed_identifiers[:2]),
            "entity_type": node.entity.entity_type.value,
            "signal": node.signal_source.value,
            "is_causal": node.is_causal,
            "context_score": round(node.context_score or 0.0, 3),
            "uncertainty": {
                "nonconformity": node.uncertainty.nonconformity_score,
                "bert_f1": node.uncertainty.bert_f1_score,
            },
            "content": {
                k: v for k, v in node.context.items()
                if not k.startswith("_")  # Exclude internal metadata
            },
        }


# Allow forward reference for Any type in admit_candidate
from typing import Any  # noqa: E402
