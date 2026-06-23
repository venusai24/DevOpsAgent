"""
Unit Tests — Supermartingale & PASC (Module 1.4).

Tests the e-process supermartingale update, escalation detection,
and PASC coverage validation. All pure math — no external services.
"""
from __future__ import annotations

import pytest

from airs.risk.pasc import (
    compute_pasc_for_tier3,
    compute_pasc_joint_max,
    validate_pasc_coverage,
)
from airs.risk.supermartingale import (
    check_escalation,
    compute_e_value,
    update_supermartingale,
)


# ─── E-value ──────────────────────────────────────────────────────────────────

class TestComputeEValue:
    def test_conforming_evidence_gives_e_below_one(self) -> None:
        # s_k = 0.20 < q̂ = 0.50 → e_k = 0.40 < 1.0
        e = compute_e_value(nonconformity_score=0.20, calibration_quantile=0.50)
        assert e == pytest.approx(0.40, abs=1e-6)
        assert e < 1.0

    def test_exact_threshold_gives_e_of_one(self) -> None:
        e = compute_e_value(nonconformity_score=0.50, calibration_quantile=0.50)
        assert e == pytest.approx(1.0, abs=1e-6)

    def test_nonconforming_gives_e_above_one(self) -> None:
        # s_k = 0.80 > q̂ = 0.50 → e_k = 1.6 > 1.0
        e = compute_e_value(nonconformity_score=0.80, calibration_quantile=0.50)
        assert e == pytest.approx(1.60, abs=1e-6)
        assert e > 1.0

    def test_zero_quantile_uses_minimum(self) -> None:
        # Zero quantile should not cause division by zero
        e = compute_e_value(nonconformity_score=0.5, calibration_quantile=0.0)
        assert 0.0 <= e <= 100.0  # Clamped

    def test_e_value_capped_at_100(self) -> None:
        # Extremely small quantile → cap e_value at 100
        e = compute_e_value(nonconformity_score=1.0, calibration_quantile=0.001)
        assert e == 100.0


# ─── Supermartingale Update ───────────────────────────────────────────────────

class TestUpdateSupermartingale:
    def test_initial_value_unchanged_by_conforming(self) -> None:
        # Conforming hop: e_k < 1 → M decreases
        M = update_supermartingale(current_M=1.0, e_value=0.5)
        assert M == pytest.approx(0.5)

    def test_nonconforming_hop_increases_M(self) -> None:
        M = update_supermartingale(current_M=1.0, e_value=2.0)
        assert M == pytest.approx(2.0)

    def test_cumulative_growth(self) -> None:
        M = 1.0
        for _ in range(5):
            M = update_supermartingale(M, e_value=2.0)
        assert M == pytest.approx(32.0, abs=1e-6)  # 2^5

    def test_cumulative_decay_with_good_evidence(self) -> None:
        M = 10.0
        for _ in range(10):
            M = update_supermartingale(M, e_value=0.5)
        # 10 * 0.5^10 ≈ 0.0098
        assert M < 0.02

    def test_negative_M_raises(self) -> None:
        with pytest.raises(ValueError):
            update_supermartingale(current_M=-1.0, e_value=1.5)


# ─── Escalation Check ─────────────────────────────────────────────────────────

class TestCheckEscalation:
    def test_below_threshold_no_escalation(self) -> None:
        # Threshold = 1/0.05 = 20.0 → M_t=15 < 20 → no escalation
        assert check_escalation(15.0, delta=0.05) is False

    def test_at_threshold_triggers_escalation(self) -> None:
        # M_t = 20.0 == 1/0.05 → escalate
        assert check_escalation(20.0, delta=0.05) is True

    def test_well_above_threshold(self) -> None:
        assert check_escalation(100.0, delta=0.05) is True

    def test_fresh_investigation_no_escalation(self) -> None:
        # M_t starts at 1.0, well below threshold
        assert check_escalation(1.0, delta=0.05) is False

    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            check_escalation(10.0, delta=0.0)

    def test_full_investigation_scenario(self) -> None:
        """Simulate 8 hops: first 5 fine, last 3 with hallucinations."""
        M = 1.0
        q_hat = 0.50
        delta = 0.05

        # 5 conforming hops (s_k = 0.30 < q̂)
        for _ in range(5):
            from airs.risk.supermartingale import compute_e_value
            e = compute_e_value(0.30, q_hat)  # e = 0.6
            M = update_supermartingale(M, e)
        assert not check_escalation(M, delta)  # Should be fine

        # 3 highly nonconforming hops (s_k = 0.95 >> q̂)
        for _ in range(3):
            from airs.risk.supermartingale import compute_e_value
            e = compute_e_value(0.95, q_hat)  # e = 1.9
            M = update_supermartingale(M, e)

        # M should now be elevated; check if escalation is triggered
        # After 5 decays (0.6^5 ≈ 0.078) then 3 growths (1.9^3 ≈ 6.86)
        # M ≈ 0.078 * 6.86 ≈ 0.535 — below threshold of 20
        assert M < 20.0  # Not yet at threshold


# ─── PASC ─────────────────────────────────────────────────────────────────────

class TestPASC:
    def test_joint_max_empty(self) -> None:
        assert compute_pasc_joint_max([]) == 0.0

    def test_joint_max_correct(self) -> None:
        scores = [0.10, 0.45, 0.30, 0.20]
        assert compute_pasc_joint_max(scores) == pytest.approx(0.45)

    def test_coverage_met(self) -> None:
        # s_max=0.40 ≤ q̂=0.50 → coverage met
        assert validate_pasc_coverage(0.40, 0.50) is True

    def test_coverage_violated(self) -> None:
        # s_max=0.60 > q̂=0.50 → violation
        assert validate_pasc_coverage(0.60, 0.50) is False

    def test_coverage_at_boundary(self) -> None:
        # s_max == q̂ → coverage met (≤ not <)
        assert validate_pasc_coverage(0.50, 0.50) is True

    def test_combined_function(self) -> None:
        scores = [0.10, 0.25, 0.35]
        s_max, met = compute_pasc_for_tier3(scores, calibration_quantile=0.50)
        assert s_max == pytest.approx(0.35)
        assert met is True

    def test_combined_function_violation(self) -> None:
        scores = [0.10, 0.65, 0.35]
        s_max, met = compute_pasc_for_tier3(scores, calibration_quantile=0.50)
        assert s_max == pytest.approx(0.65)
        assert met is False

    def test_invalid_score_in_joint_max_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_pasc_joint_max([0.2, 1.5])
