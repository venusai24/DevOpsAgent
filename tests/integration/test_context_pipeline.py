"""
Integration Tests — Context Pipeline (Module 1.12).

Tests the full candidate → admission → scoring → eviction pipeline
using realistic synthetic evidence. No external services required.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from airs.context.manager import ProactiveContextManager
from airs.context.scorer import ContextScorer
from airs.models.evidence import EvidenceCandidate, EntityType, SignalSource


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_candidate(
    service: str = "payment-service",
    signal_source: SignalSource = SignalSource.METRICS,
    entity_type: EntityType = EntityType.SERVICE,
    token_count: int = 200,
    hop_index: int = 1,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        entity_type=entity_type,
        raw_identifiers=[service],
        signal_source=signal_source,
        filtered_content={
            "metric_name": "http_request_duration_seconds",
            "anomalous_points": [{"timestamp": 1718000120.0, "value": 2.5}],
            "summary_stats": {"mean": 0.12, "p99": 2.5, "count": 5},
            "_token_count": token_count,
        },
        mcp_server_id="prometheus",
        tool_invoked="execute_range_query",
        query_parameters_hash="abc123",
        raw_response_digest="def456",
        idempotency_key=f"idem:{service}:{hop_index}",
        observed_at=datetime.now(timezone.utc),
        valid_from=datetime.now(timezone.utc),
        valid_until=None,
        hop_index=hop_index,
        token_count=token_count,
    )


class MockEntityResolutionEngine:
    """Minimal mock that returns a deterministic GUID per service name."""
    def resolve_entity(self, entity_type, identifiers):
        name = identifiers[0] if identifiers else "unknown"
        return uuid.uuid5(uuid.NAMESPACE_DNS, name)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestCandidateAdmission:
    def test_admit_single_candidate(self, initial_investigation_state):
        """Single candidate is admitted to active tier."""
        manager = ProactiveContextManager.from_settings()
        er = MockEntityResolutionEngine()
        candidate = make_candidate(token_count=200)

        new_state = manager.admit_candidate(
            state=initial_investigation_state,
            candidate=candidate,
            entity_resolution_engine=er,
        )
        assert len(new_state.insight_tiers.active) == 1

    def test_token_headroom_prevents_excess_admission(self, initial_investigation_state):
        """Admission is blocked when active evidence partition is full."""
        manager = ProactiveContextManager.from_settings()
        er = MockEntityResolutionEngine()

        # Fill the partition by setting usage to near capacity
        from airs.context.budget import BudgetTracker
        budget = initial_investigation_state.context_budget

        # Create candidate that would overfill (token_count > remaining headroom)
        massive_candidate = make_candidate(
            token_count=budget.active_evidence_tokens + 1000,
        )
        new_state = manager.admit_candidate(
            state=initial_investigation_state,
            candidate=massive_candidate,
            entity_resolution_engine=er,
        )
        # Should be discarded
        assert len(new_state.insight_tiers.active) == 0
        assert new_state.insight_tiers.discarded_count == 1

    def test_entity_resolution_produces_canonical_guid(self, initial_investigation_state):
        """Same service name → same canonical GUID on both admissions."""
        manager = ProactiveContextManager.from_settings()
        er = MockEntityResolutionEngine()
        c1 = make_candidate(service="payment-service", hop_index=1)
        c2 = make_candidate(service="payment-service", hop_index=2, signal_source=SignalSource.LOGS)

        state1 = manager.admit_candidate(initial_investigation_state, c1, er)
        state2 = manager.admit_candidate(state1, c2, er)

        guids = [
            node.entity.canonical_guid
            for node in state2.insight_tiers.active
        ]
        # Both nodes should have the same canonical GUID (same service)
        assert guids[0] == guids[1]


class TestEvictionPipeline:
    def test_no_eviction_below_pressure(self, initial_investigation_state):
        """No eviction when utilisation < 85%."""
        manager = ProactiveContextManager.from_settings()
        er = MockEntityResolutionEngine()

        # Admit a small candidate (< 85% pressure)
        candidate = make_candidate(token_count=500)
        state = manager.admit_candidate(initial_investigation_state, candidate, er)
        new_state, metrics = manager.evaluate_and_evict(state)

        assert len(new_state.insight_tiers.active) == 1
        assert metrics.eviction_count == 0

    def test_causal_nodes_never_evicted(self, initial_investigation_state):
        """Causal chain nodes are immune to eviction."""
        manager = ProactiveContextManager.from_settings()
        er = MockEntityResolutionEngine()

        candidate = make_candidate(token_count=500)
        state = manager.admit_candidate(initial_investigation_state, candidate, er)

        # Mark the admitted node as causal
        node = state.insight_tiers.active[0]
        causal_node = node.model_copy(update={"is_causal": True})
        updated_active = [causal_node]
        state = state.model_copy(
            update={"insight_tiers": state.insight_tiers.model_copy(update={"active": updated_active})}
        )
        # Mark in graph too
        state.graph.causal_chain_ids.append(causal_node.node_id)

        new_state, metrics = manager.evaluate_and_evict(state)
        # Causal node should NOT be evicted
        assert any(n.node_id == causal_node.node_id for n in new_state.insight_tiers.active)


class TestContextWindow:
    def test_context_window_structure(self, initial_investigation_state):
        """Context window has expected keys."""
        manager = ProactiveContextManager.from_settings()
        ctx = manager.get_context_window(initial_investigation_state)

        assert "causal_chain" in ctx
        assert "active_evidence" in ctx
        assert "summarized_evidence" in ctx
        assert "total_active" in ctx

    def test_context_window_with_evidence(self, initial_investigation_state):
        """Context window includes admitted evidence."""
        manager = ProactiveContextManager.from_settings()
        er = MockEntityResolutionEngine()
        candidate = make_candidate(token_count=300)
        state = manager.admit_candidate(initial_investigation_state, candidate, er)

        ctx = manager.get_context_window(state)
        assert ctx["total_active"] == 1


class TestContextScorer:
    def test_recency_decay_favors_recent_nodes(self, initial_investigation_state):
        """More recent nodes should have higher recency scores."""
        from airs.models.evidence import (
            EntityDescriptor, EvidenceNode, ProvenanceRecord,
            TemporalityBound, UncertaintyMetrics
        )
        from datetime import timedelta

        scorer = ContextScorer()

        def make_node(seconds_ago: int, nid: str) -> EvidenceNode:
            t = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
            return EvidenceNode(
                node_id=nid,
                entity=EntityDescriptor(
                    canonical_guid=uuid.uuid4(),
                    entity_type=EntityType.SERVICE,
                    observed_identifiers=["svc"],
                ),
                signal_source=SignalSource.METRICS,
                context={},
                uncertainty=UncertaintyMetrics(
                    nonconformity_score=0.3,
                    calibration_quantile=0.5,
                    li_score=0.3,
                    bert_f1_score=0.8,
                ),
                provenance=ProvenanceRecord(
                    mcp_server_id="prometheus",
                    tool_invoked="execute_query",
                    query_parameters_hash="hash",
                    raw_response_digest="digest",
                    idempotency_key="idem",
                ),
                temporality=TemporalityBound(
                    observed_at=t, valid_from=t, valid_until=None
                ),
                hop_index=1,
                is_causal=False,
            )

        recent_node = make_node(10, "recent")
        old_node = make_node(3600, "old")

        recent_score = scorer.score_node(recent_node, initial_investigation_state.graph, 1)
        old_score = scorer.score_node(old_node, initial_investigation_state.graph, 1)

        assert recent_score.recency > old_score.recency
