"""
Unit Tests — Data Models (Module 1.2).

Tests every model for correct field defaults, validation constraints,
computed properties, and cross-model composition in InvestigationState.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from airs.models import (
    AlertPayload,
    ContextBudget,
    ContextScore,
    DirectedEdge,
    EdgeType,
    EntityDescriptor,
    EntityType,
    EvidenceCandidate,
    EvidenceNode,
    ExecutionIntent,
    Hypothesis,
    HypothesisStatus,
    InformationPursuitState,
    InsightTier,
    InsightTiers,
    IntentAction,
    InvestigationGraph,
    InvestigationState,
    OpState,
    PlaybookQuery,
    ProvenanceRecord,
    SignalSource,
    SignalType,
    StepRiskEntry,
    TemporalityBound,
    ToolSpec,
    ToolTier,
    TrajectoryRiskState,
    UncertaintyMetrics,
    VerificationResult,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def now() -> datetime:
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def sample_uncertainty() -> UncertaintyMetrics:
    return UncertaintyMetrics(
        nonconformity_score=0.25,
        calibration_quantile=0.50,
        li_score=0.70,
        bert_f1_score=0.85,
    )


@pytest.fixture
def sample_provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        mcp_server_id="prometheus",
        tool_invoked="execute_range_query",
        query_parameters_hash="abc123",
        raw_response_digest="def456",
        idempotency_key="inv-001:hop-1:execute_range_query:abc123",
    )


@pytest.fixture
def sample_temporality(now: datetime) -> TemporalityBound:
    return TemporalityBound(
        observed_at=now,
        valid_from=now - timedelta(minutes=15),
        valid_until=now,
    )


@pytest.fixture
def sample_entity() -> EntityDescriptor:
    return EntityDescriptor(
        canonical_guid=uuid.uuid5(uuid.NAMESPACE_DNS, "frontend-service"),
        entity_type=EntityType.SERVICE,
        observed_identifiers=["frontend", "frontend-service", "frontend:8080"],
    )


@pytest.fixture
def sample_evidence_node(
    sample_entity: EntityDescriptor,
    sample_uncertainty: UncertaintyMetrics,
    sample_provenance: ProvenanceRecord,
    sample_temporality: TemporalityBound,
) -> EvidenceNode:
    return EvidenceNode(
        node_id="node-001",
        entity=sample_entity,
        signal_source=SignalSource.METRICS,
        context={
            "metric": "http_requests_total",
            "error_rate": 0.42,
            "z_score": 5.3,
        },
        uncertainty=sample_uncertainty,
        provenance=sample_provenance,
        temporality=sample_temporality,
        hop_index=1,
        is_causal=False,
    )


@pytest.fixture
def sample_alert(now: datetime) -> AlertPayload:
    return AlertPayload(
        alert_id="alert-001",
        alert_name="HighErrorRate",
        severity="critical",
        service="frontend",
        namespace="production",
        description="HTTP error rate > 40%",
        fired_at=now,
    )


# ─── UncertaintyMetrics ───────────────────────────────────────────────────────

class TestUncertaintyMetrics:
    def test_valid_creation(self, sample_uncertainty: UncertaintyMetrics) -> None:
        assert 0.0 <= sample_uncertainty.nonconformity_score <= 1.0
        assert 0.0 <= sample_uncertainty.bert_f1_score <= 1.0

    def test_bounds_validation(self) -> None:
        with pytest.raises(Exception):
            UncertaintyMetrics(
                nonconformity_score=1.5,  # > 1.0 — invalid
                calibration_quantile=0.5,
                li_score=0.5,
                bert_f1_score=0.5,
            )


# ─── EntityDescriptor ─────────────────────────────────────────────────────────

class TestEntityDescriptor:
    def test_uuid_serialization(self, sample_entity: EntityDescriptor) -> None:
        data = sample_entity.model_dump()
        assert isinstance(data["canonical_guid"], str)

    def test_entity_type_enum(self, sample_entity: EntityDescriptor) -> None:
        assert sample_entity.entity_type == EntityType.SERVICE


# ─── EvidenceNode ─────────────────────────────────────────────────────────────

class TestEvidenceNode:
    def test_defaults(self, sample_evidence_node: EvidenceNode) -> None:
        assert sample_evidence_node.is_causal is False
        assert sample_evidence_node.context_score is None
        assert sample_evidence_node.hop_index == 1

    def test_round_trip_json(self, sample_evidence_node: EvidenceNode) -> None:
        dumped = sample_evidence_node.model_dump_json()
        restored = EvidenceNode.model_validate_json(dumped)
        assert restored.node_id == sample_evidence_node.node_id
        assert restored.signal_source == SignalSource.METRICS


# ─── DirectedEdge ─────────────────────────────────────────────────────────────

class TestDirectedEdge:
    def test_edge_creation(self) -> None:
        edge = DirectedEdge(
            source_node_id="node-001",
            target_node_id="node-002",
            edge_type=EdgeType.CAUSED_BY,
            confidence=0.90,
        )
        assert edge.edge_type == EdgeType.CAUSED_BY
        assert edge.metadata == {}


# ─── TrajectoryRiskState ──────────────────────────────────────────────────────

class TestTrajectoryRiskState:
    def test_escalation_threshold(self) -> None:
        state = TrajectoryRiskState(delta_alarm=0.05)
        assert state.escalation_threshold == 20.0

    def test_should_not_escalate_initially(self) -> None:
        state = TrajectoryRiskState()
        assert state.should_escalate is False  # M_t starts at 1.0, threshold = 20

    def test_should_escalate_when_threshold_breached(self) -> None:
        state = TrajectoryRiskState(supermartingale_value=25.0, delta_alarm=0.05)
        assert state.should_escalate is True


# ─── InformationPursuitState ──────────────────────────────────────────────────

class TestInformationPursuitState:
    def test_plateau_detection(self) -> None:
        state = InformationPursuitState(consecutive_low_delta=3)
        assert state.is_plateau is True

    def test_no_plateau_initially(self) -> None:
        state = InformationPursuitState()
        assert state.is_plateau is False

    def test_complete_when_below_epsilon(self) -> None:
        state = InformationPursuitState(
            current_missing_mass=0.03,
            epsilon_threshold=0.05,
        )
        assert state.is_complete is True


# ─── ContextBudget ────────────────────────────────────────────────────────────

class TestContextBudget:
    def test_token_ceilings(self) -> None:
        budget = ContextBudget(max_tokens=80_000)
        assert budget.constitution_tokens == 6_400     # 8%
        assert budget.active_evidence_tokens == 40_000  # 50%
        assert budget.playbook_tokens == 8_000          # 10%
        assert budget.causal_chain_tokens == 24_000     # 60% of active_evidence

    def test_custom_budget(self) -> None:
        budget = ContextBudget(max_tokens=40_000)
        assert budget.active_evidence_tokens == 20_000


# ─── ContextScore ─────────────────────────────────────────────────────────────

class TestContextScore:
    def test_composite_calculation(self) -> None:
        score = ContextScore(
            relevance=1.0,
            recency=1.0,
            causal_importance=1.0,
            uniqueness=1.0,
            diagnostic_value=1.0,
        )
        assert abs(score.composite - 1.0) < 1e-6

    def test_zero_composite(self) -> None:
        score = ContextScore(
            relevance=0.0,
            recency=0.0,
            causal_importance=0.0,
            uniqueness=0.0,
            diagnostic_value=0.0,
        )
        assert score.composite == 0.0

    def test_weights_sum_to_one(self) -> None:
        total = sum(ContextScore.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6


# ─── Hypothesis ───────────────────────────────────────────────────────────────

class TestHypothesis:
    def test_creation(self) -> None:
        h = Hypothesis(
            hypothesis_id="h-001",
            statement="frontend service is experiencing connection pool exhaustion",
            created_at_hop=1,
            last_updated_hop=1,
        )
        assert h.status == HypothesisStatus.ACTIVE
        assert h.confidence == 0.5

    def test_record_confidence(self) -> None:
        h = Hypothesis(
            hypothesis_id="h-001",
            statement="test",
            created_at_hop=0,
            last_updated_hop=0,
        )
        h.record_confidence(hop_index=2, new_confidence=0.85)
        assert h.confidence == 0.85
        assert h.last_updated_hop == 2
        assert len(h.confidence_history) == 1


# ─── ExecutionIntent ──────────────────────────────────────────────────────────

class TestExecutionIntent:
    def test_execute_tool_valid(self) -> None:
        intent = ExecutionIntent(
            action=IntentAction.EXECUTE_TOOL,
            tool_spec=ToolSpec(
                tool_name="execute_range_query",
                mcp_server_id="prometheus",
                tier=ToolTier.OBSERVATION,
                signal_type=SignalType.METRICS,
            ),
        )
        assert intent.action == IntentAction.EXECUTE_TOOL

    def test_execute_tool_missing_spec_raises(self) -> None:
        with pytest.raises(Exception):
            ExecutionIntent(action=IntentAction.EXECUTE_TOOL)  # no tool_spec

    def test_query_playbook_valid(self) -> None:
        intent = ExecutionIntent(
            action=IntentAction.QUERY_PLAYBOOK,
            playbook_query=PlaybookQuery(
                collection="diagnostic_knowledge",
                query_text="connection pool exhaustion symptoms",
            ),
        )
        assert intent.playbook_query is not None

    def test_diagnose_needs_no_payload(self) -> None:
        intent = ExecutionIntent(action=IntentAction.DIAGNOSE)
        assert intent.tool_spec is None
        assert intent.playbook_query is None


# ─── InvestigationGraph ───────────────────────────────────────────────────────

class TestInvestigationGraph:
    def test_add_and_retrieve_node(self, sample_evidence_node: EvidenceNode) -> None:
        graph = InvestigationGraph()
        graph.add_node(sample_evidence_node)
        assert graph.node_count() == 1
        assert graph.get_node("node-001") is sample_evidence_node

    def test_add_edge(self) -> None:
        graph = InvestigationGraph()
        edge = DirectedEdge(
            source_node_id="node-001",
            target_node_id="node-002",
            edge_type=EdgeType.CAUSED_BY,
            confidence=0.80,
        )
        graph.add_edge(edge)
        assert graph.edge_count() == 1

    def test_causal_chain(self, sample_evidence_node: EvidenceNode) -> None:
        graph = InvestigationGraph()
        graph.add_node(sample_evidence_node)
        graph.causal_chain_ids = ["node-001"]
        chain = graph.get_causal_chain()
        assert len(chain) == 1
        assert chain[0].node_id == "node-001"


# ─── InvestigationState ───────────────────────────────────────────────────────

class TestInvestigationState:
    def test_creation_with_defaults(self, sample_alert: AlertPayload) -> None:
        state = InvestigationState(
            investigation_id="inv-001",
            workflow_run_id="run-abc",
            alert=sample_alert,
        )
        assert state.total_hop_count == 0
        assert state.leading_hypothesis_id is None
        assert state.graph.node_count() == 0
        assert state.risk_state.supermartingale_value == 1.0
        assert state.pursuit_state.current_missing_mass == 1.0

    def test_increment_hop(self, sample_alert: AlertPayload) -> None:
        state = InvestigationState(
            investigation_id="inv-001",
            workflow_run_id="run-abc",
            alert=sample_alert,
        )
        new_state = state.increment_hop()
        assert new_state.total_hop_count == 1
        assert state.total_hop_count == 0  # Original unchanged

    def test_get_active_hypotheses_empty(self, sample_alert: AlertPayload) -> None:
        state = InvestigationState(
            investigation_id="inv-001",
            workflow_run_id="run-abc",
            alert=sample_alert,
        )
        assert state.get_active_hypotheses() == []

    def test_json_round_trip(self, sample_alert: AlertPayload) -> None:
        state = InvestigationState(
            investigation_id="inv-001",
            workflow_run_id="run-abc",
            alert=sample_alert,
        )
        dumped = state.model_dump_json()
        restored = InvestigationState.model_validate_json(dumped)
        assert restored.investigation_id == "inv-001"
        assert restored.alert.alert_name == "HighErrorRate"
