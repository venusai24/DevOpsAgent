"""
Risk Models.

Data structures for the Conformal Risk Control Engine:
trajectory risk, supermartingale tracking, step risk, and PASC verification.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StepRiskEntry(BaseModel):
    """
    Risk record for a single investigation hop.

    hop_index:         The hop number this record belongs to.
    nonconformity_score: Raw s_k = w1*LI + w2*(1-BERT_F1).
    step_risk_score:   Sigmoid-normalised risk ∈ [0, 1].
    e_value:           e_k = s_k / q_hat_{1-alpha} — the e-process value.
    calibration_quantile: q_hat used for this step's computation.
    """
    hop_index: int = Field(ge=0)
    nonconformity_score: float = Field(ge=0.0, le=1.0)
    step_risk_score: float = Field(ge=0.0, le=1.0)
    e_value: float = Field(ge=0.0, description="e-process contribution to supermartingale")
    calibration_quantile: float = Field(ge=0.0, le=1.0)


class TrajectoryRiskState(BaseModel):
    """
    Complete risk tracking state for an ongoing investigation.

    trajectory_risk_score: Noisy-or aggregation of all step risks R_traj ∈ [0,1].
    supermartingale_value: Running product M_t = Π e_k. Escalate when M_t >= 1/δ.
    lambda_threshold:      Per-step risk ceiling. ABSTAIN/PRUNE if step_risk > λ.
    alpha_coverage:        Conformal coverage level (e.g., 0.10 for 90% coverage).
    delta_alarm:           Escalation trigger: 1/δ is the M_t ceiling.
    step_risks:            Full audit trail of per-hop risk entries.
    pasc_joint_max_score:  Highest PASC score seen across all Tier 3 tool validations.
    """
    trajectory_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supermartingale_value: float = Field(default=1.0, ge=0.0)
    lambda_threshold: float = Field(default=0.30, gt=0.0, le=1.0)
    alpha_coverage: float = Field(default=0.10, gt=0.0, lt=1.0)
    delta_alarm: float = Field(default=0.05, gt=0.0, lt=1.0)
    step_risks: list[StepRiskEntry] = Field(default_factory=list)
    pasc_joint_max_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def escalation_threshold(self) -> float:
        """M_t >= 1/delta triggers escalation."""
        return 1.0 / self.delta_alarm

    @property
    def should_escalate(self) -> bool:
        return self.supermartingale_value >= self.escalation_threshold

    @property
    def expected_trajectory_risk(self) -> float:
        """E[R^traj] = trajectory_risk_score (noisy-or aggregate)."""
        return self.trajectory_risk_score
