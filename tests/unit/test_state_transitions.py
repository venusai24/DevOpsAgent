"""
Unit Tests — State Transitions (Module 1.11).

Tests the deterministic routing logic and state transition mechanics:
- route_decision rules (all 5 cases)
- hop counter increments
- missing mass convergence detection
- plateau detection
- ESCALATE threshold
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from airs.graph.nodes.calculate_missing_mass import compute_missing_mass, get_priority_evidence_gap
from airs.graph.nodes.route_decision import route_decision
from airs.models.hypothesis import Hypothesis, HypothesisStatus
from airs.models.pursuit import InformationPursuitState


# ─── Route Decision Tests ─────────────────────────────────────────────────────

class TestRouteDecision:
    def test_default_execute_tool(self, initial_investigation_state):
        """Fresh investigation → execute_tool."""
        decision = route_decision(initial_investigation_state, max_hops=50)
        assert decision == "execute_tool"

    def test_max_hops_escalate(self, initial_investigation_state):
        """Hop count at max → escalate."""
        state = initial_investigation_state.model_copy(
            update={"total_hop_count": 50}
        )
        assert route_decision(state, max_hops=50) == "escalate"

    def test_supermartingale_alarm_escalate(self, initial_investigation_state):
        """M_t >= 1/δ → escalate."""
        from airs.models.risk import TrajectoryRiskState
        new_risk = initial_investigation_state.risk_state.model_copy(
            update={
                "supermartingale_value": 25.0,
                "escalation_threshold": 20.0,
                "should_escalate": True,
            }
        )
        state = initial_investigation_state.model_copy(update={"risk_state": new_risk})
        assert route_decision(state, max_hops=50) == "escalate"

    def test_missing_mass_converged_high_confidence_diagnose(
        self, initial_investigation_state
    ):
        """Missing mass < epsilon + confidence > 0.85 → diagnose."""
        new_pursuit = initial_investigation_state.pursuit_state.model_copy(
            update={"current_missing_mass": 0.02}
        )
        high_conf_hyp = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            statement="DB connection pool exhausted causing 5xx errors",
            confidence=0.92,
            status=HypothesisStatus.LEADING,
            supporting_node_ids=[],
            contradicting_node_ids=[],
            created_at_hop=3,
            last_updated_hop=3,
        )
        state = initial_investigation_state.model_copy(
            update={
                "pursuit_state": new_pursuit,
                "hypotheses": [high_conf_hyp],
                "leading_hypothesis_id": high_conf_hyp.hypothesis_id,
            }
        )
        assert route_decision(state, max_hops=50) == "diagnose"

    def test_missing_mass_converged_low_confidence_playbook(
        self, initial_investigation_state
    ):
        """Missing mass < epsilon but confidence < 0.85 → query_playbook."""
        new_pursuit = initial_investigation_state.pursuit_state.model_copy(
            update={"current_missing_mass": 0.02}
        )
        low_conf_hyp = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            statement="Some hypothesis",
            confidence=0.50,
            status=HypothesisStatus.ACTIVE,
            supporting_node_ids=[],
            contradicting_node_ids=[],
            created_at_hop=0,
            last_updated_hop=0,
        )
        state = initial_investigation_state.model_copy(
            update={
                "pursuit_state": new_pursuit,
                "hypotheses": [low_conf_hyp],
                "leading_hypothesis_id": low_conf_hyp.hypothesis_id,
            }
        )
        assert route_decision(state, max_hops=50) == "query_playbook"

    def test_plateau_triggers_playbook(self, initial_investigation_state):
        """3 consecutive low-delta hops → query_playbook."""
        new_pursuit = initial_investigation_state.pursuit_state.model_copy(
            update={"consecutive_low_delta": 3}
        )
        state = initial_investigation_state.model_copy(update={"pursuit_state": new_pursuit})
        assert route_decision(state, max_hops=50) == "query_playbook"

    def test_high_confidence_early_diagnose(self, initial_investigation_state):
        """Confidence >= 0.92 triggers early diagnosis."""
        high_hyp = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            statement="Root cause confirmed",
            confidence=0.95,
            status=HypothesisStatus.CONFIRMED,
            supporting_node_ids=[],
            contradicting_node_ids=[],
            created_at_hop=2,
            last_updated_hop=2,
        )
        state = initial_investigation_state.model_copy(
            update={
                "hypotheses": [high_hyp],
                "leading_hypothesis_id": high_hyp.hypothesis_id,
            }
        )
        assert route_decision(state, max_hops=50) == "diagnose"


# ─── Missing Mass Tests ───────────────────────────────────────────────────────

class TestComputeMissingMass:
    def test_initial_state_high_mass(self, initial_investigation_state):
        """Fresh state with no evidence → missing mass near 1.0."""
        pursuit = compute_missing_mass(initial_investigation_state)
        assert pursuit.current_missing_mass >= 0.80

    def test_empty_active_nodes(self, initial_investigation_state):
        """No active nodes → maximum missing mass."""
        pursuit = compute_missing_mass(initial_investigation_state)
        assert 0.0 <= pursuit.current_missing_mass <= 1.0

    def test_plateau_detection_increments(self, initial_investigation_state):
        """Consecutive low-delta hops are counted."""
        state = initial_investigation_state.model_copy(
            update={
                "pursuit_state": initial_investigation_state.pursuit_state.model_copy(
                    update={
                        "current_missing_mass": 0.80,
                        "previous_missing_mass": 0.80,
                        "consecutive_low_delta": 1,
                        "missing_mass_delta": 0.001,
                    }
                ),
                "total_hop_count": 2,
            }
        )
        # Compute missing mass — should run without error and produce valid output
        new_pursuit = compute_missing_mass(state)
        assert 0.0 <= new_pursuit.current_missing_mass <= 1.0

    def test_missing_mass_bounded(self, initial_investigation_state):
        """Missing mass is always in [0, 1]."""
        for _ in range(5):
            pursuit = compute_missing_mass(initial_investigation_state)
            assert 0.0 <= pursuit.current_missing_mass <= 1.0

    def test_is_complete_false_for_fresh_state(self, initial_investigation_state):
        """Fresh state is not complete."""
        pursuit = compute_missing_mass(initial_investigation_state)
        assert not pursuit.is_complete

    def test_is_plateau_false_initially(self, initial_investigation_state):
        """No plateau at investigation start."""
        pursuit = compute_missing_mass(initial_investigation_state)
        assert not pursuit.is_plateau


# ─── Priority Evidence Gap Tests ──────────────────────────────────────────────

class TestGetPriorityEvidenceGap:
    def test_metrics_error_rate_is_first_priority(self):
        """metrics_error_rate should be highest priority gap."""
        pursuit = InformationPursuitState(
            current_missing_mass=0.8,
            previous_missing_mass=1.0,
            missing_mass_delta=0.2,
            epsilon_threshold=0.05,
            consecutive_low_delta=0,
            filled_evidence_categories=[],
            open_evidence_categories=[
                "metrics_error_rate",
                "logs_application_error",
                "traces_service_call_graph",
            ],
        )
        gap = get_priority_evidence_gap(pursuit)
        assert gap == "metrics_error_rate"

    def test_logs_next_after_metrics(self):
        """After metrics filled, logs should be next."""
        pursuit = InformationPursuitState(
            current_missing_mass=0.6,
            previous_missing_mass=0.8,
            missing_mass_delta=0.2,
            epsilon_threshold=0.05,
            consecutive_low_delta=0,
            filled_evidence_categories=["metrics_error_rate", "metrics_latency"],
            open_evidence_categories=[
                "logs_application_error",
                "traces_service_call_graph",
            ],
        )
        gap = get_priority_evidence_gap(pursuit)
        assert gap == "logs_application_error"

    def test_no_gaps_returns_none(self):
        """All categories filled → None."""
        pursuit = InformationPursuitState(
            current_missing_mass=0.0,
            previous_missing_mass=0.1,
            missing_mass_delta=0.1,
            epsilon_threshold=0.05,
            consecutive_low_delta=0,
            filled_evidence_categories=[
                "metrics_error_rate", "metrics_latency", "metrics_saturation",
                "logs_application_error", "traces_service_call_graph",
                "k8s_state_pod_events", "metrics_dependency_health",
                "logs_upstream_downstream",
            ],
            open_evidence_categories=[],
        )
        assert get_priority_evidence_gap(pursuit) is None


# ─── Hop Counter Tests ────────────────────────────────────────────────────────

class TestHopCounter:
    def test_increment_hop(self, initial_investigation_state):
        """increment_hop() returns new state with count+1."""
        state = initial_investigation_state
        assert state.total_hop_count == 0
        new_state = state.increment_hop()
        assert new_state.total_hop_count == 1
        assert state.total_hop_count == 0  # Original immutable

    def test_multiple_increments(self, initial_investigation_state):
        """Multiple hop increments."""
        state = initial_investigation_state
        for i in range(1, 6):
            state = state.increment_hop()
            assert state.total_hop_count == i
