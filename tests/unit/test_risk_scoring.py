"""
Unit Tests — Conformal Risk Scoring (Module 1.4).

Tests trajectory risk, step risk, combined nonconformity scores,
and CalibrationStore loading. No external services required.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from airs.risk.calibration import CalibrationStore
from airs.risk.nonconformity import (
    compute_combined_nonconformity,
    compute_token_entropy_proxy,
)
from airs.risk.trajectory_risk import compute_step_risk, compute_trajectory_risk


# ─── Token Entropy Proxy ──────────────────────────────────────────────────────

class TestTokenEntropyProxy:
    def test_empty_logprobs_returns_maximum_uncertainty(self) -> None:
        assert compute_token_entropy_proxy([]) == 1.0

    def test_perfectly_confident_token_returns_low_entropy(self) -> None:
        # logprob = 0.0 → p = 1.0 → H = 0
        result = compute_token_entropy_proxy([0.0])
        assert result == 0.0

    def test_uniform_uncertainty_returns_nonzero(self) -> None:
        # logprob = log(1/vocab) for each token → non-zero entropy
        import math
        # Use a very small log-prob to simulate high uncertainty
        lp = math.log(1e-10)   # Very low probability → high per-token entropy
        result = compute_token_entropy_proxy([lp] * 20)
        assert result > 0.0    # Must be non-zero for uncertain tokens
        assert result <= 1.0   # Must be normalised


    def test_result_always_in_unit_interval(self) -> None:
        logprobs = [-0.1, -0.5, -1.0, -3.0, -10.0]
        result = compute_token_entropy_proxy(logprobs)
        assert 0.0 <= result <= 1.0

    def test_invalid_positive_logprobs_filtered(self) -> None:
        # Positive logprobs are invalid — should be filtered, fallback to 1.0
        result = compute_token_entropy_proxy([1.0, 2.0])
        assert result == 1.0


# ─── Combined Nonconformity Score ─────────────────────────────────────────────

class TestCombinedNonconformity:
    def test_ideal_evidence_gives_low_score(self) -> None:
        # LI=0 (confident), BERT_F1=1.0 (perfect) → s_k = 0
        s = compute_combined_nonconformity(li_score=0.0, bert_f1_score=1.0)
        assert s == pytest.approx(0.0)

    def test_worst_case_gives_high_score(self) -> None:
        # LI=1 (uncertain), BERT_F1=0 (hallucinated) → s_k = 1
        s = compute_combined_nonconformity(li_score=1.0, bert_f1_score=0.0)
        assert s == pytest.approx(1.0)

    def test_typical_case(self) -> None:
        # LI=0.3, BERT_F1=0.85, w1=0.4, w2=0.6
        # s_k = 0.4*0.3 + 0.6*(1-0.85) = 0.12 + 0.09 = 0.21
        s = compute_combined_nonconformity(
            li_score=0.3, bert_f1_score=0.85, w1=0.4, w2=0.6
        )
        assert s == pytest.approx(0.21, abs=1e-6)

    def test_invalid_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_combined_nonconformity(0.5, 0.5, w1=0.3, w2=0.3)

    def test_result_clamped_to_unit_interval(self) -> None:
        # Edge case — both at extreme
        s = compute_combined_nonconformity(0.0, 0.0, w1=0.4, w2=0.6)
        assert 0.0 <= s <= 1.0


# ─── Step Risk (sigmoid) ──────────────────────────────────────────────────────

class TestComputeStepRisk:
    def test_at_calibration_quantile_gives_half(self) -> None:
        # s_k == q̂ → sigmoid(0) = 0.5
        risk = compute_step_risk(
            nonconformity_score=0.50,
            calibration_quantile=0.50,
        )
        assert risk == pytest.approx(0.5, abs=1e-6)

    def test_well_below_threshold_gives_low_risk(self) -> None:
        risk = compute_step_risk(
            nonconformity_score=0.10,
            calibration_quantile=0.50,
        )
        assert risk < 0.05

    def test_well_above_threshold_gives_high_risk(self) -> None:
        risk = compute_step_risk(
            nonconformity_score=0.90,
            calibration_quantile=0.50,
        )
        assert risk > 0.95

    def test_result_bounded(self) -> None:
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = compute_step_risk(s, 0.50)
            assert 0.0 <= r <= 1.0

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_step_risk(nonconformity_score=1.5, calibration_quantile=0.5)

    def test_invalid_quantile_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_step_risk(nonconformity_score=0.5, calibration_quantile=-0.1)


# ─── Trajectory Risk (noisy-or) ───────────────────────────────────────────────

class TestComputeTrajectoryRisk:
    def test_empty_returns_zero(self) -> None:
        assert compute_trajectory_risk([]) == 0.0

    def test_single_step(self) -> None:
        # With one step risk of 0.5: R = 1 - (1-0.5) = 0.5
        assert compute_trajectory_risk([0.5]) == pytest.approx(0.5, abs=1e-6)

    def test_zero_risks_give_zero_trajectory(self) -> None:
        assert compute_trajectory_risk([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_any_risk_one_gives_near_one(self) -> None:
        result = compute_trajectory_risk([0.1, 1.0, 0.2])
        assert result >= 0.99  # Clamped at 0.999

    def test_monotonically_increases_with_steps(self) -> None:
        r1 = compute_trajectory_risk([0.3])
        r2 = compute_trajectory_risk([0.3, 0.3])
        r3 = compute_trajectory_risk([0.3, 0.3, 0.3])
        assert r1 < r2 < r3

    def test_result_bounded(self) -> None:
        risks = [0.3] * 20
        r = compute_trajectory_risk(risks)
        assert 0.0 <= r <= 1.0

    def test_invalid_risk_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_trajectory_risk([0.5, 1.5])


# ─── CalibrationStore ─────────────────────────────────────────────────────────

class TestCalibrationStore:
    def _make_store(self, data: dict) -> CalibrationStore:
        """Create a CalibrationStore from a temp JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = Path(f.name)
        return CalibrationStore(path)

    def test_load_defaults_from_real_file(self) -> None:
        """Smoke test: load from the real calibration/defaults.json."""
        from airs.config import settings
        store = CalibrationStore(settings.calibration_dir / "defaults.json")
        w1, w2 = store.get_nonconformity_weights()
        assert abs(w1 + w2 - 1.0) < 1e-6

    def test_get_collection_alpha(self) -> None:
        data = {
            "conann": {
                "default_alpha": 0.10,
                "collections": {
                    "diagnostic_knowledge": {"alpha": 0.08, "q_hat": 0.45, "max_expansions": 5, "typical_k": 3},
                },
            },
            "risk": {},
            "context": {},
        }
        store = self._make_store(data)
        assert store.get_collection_alpha("diagnostic_knowledge") == 0.08
        assert store.get_collection_alpha("unknown_collection") == 0.10

    def test_get_q_hat_default(self) -> None:
        store = self._make_store({"conann": {}, "risk": {}, "context": {}})
        assert store.get_q_hat("any_collection") == 0.50

    def test_nonconformity_weights_sum_to_one(self) -> None:
        store = self._make_store({
            "conann": {},
            "risk": {
                "nonconformity_weights": {"w1_li_score": 0.40, "w2_bert_f1": 0.60}
            },
            "context": {},
        })
        w1, w2 = store.get_nonconformity_weights()
        assert abs(w1 + w2 - 1.0) < 1e-6

    def test_missing_file_uses_emergency_defaults(self) -> None:
        store = CalibrationStore(Path("/nonexistent/path/calibration.json"))
        w1, w2 = store.get_nonconformity_weights()
        assert w1 == pytest.approx(0.40)
        assert w2 == pytest.approx(0.60)
