"""
Trajectory Risk Computation — Module 1.4.

Implements per-step risk sigmoid normalization and noisy-or trajectory
risk aggregation.

    step_risk_k = σ(scale * (s_k - q̂))     where σ is the sigmoid function
    R_traj = 1 - Π(1 - step_risk_k)         noisy-or aggregation
"""
from __future__ import annotations

import math
import logging

log = logging.getLogger(__name__)


def compute_step_risk(
    nonconformity_score: float,
    calibration_quantile: float,
    scale: float = 10.0,
) -> float:
    """
    Compute per-step risk from a nonconformity score using sigmoid normalization.

    Maps s_k → [0, 1] risk score centered around the calibration quantile q̂:
    - s_k == q̂  → risk = 0.5 (at threshold)
    - s_k << q̂  → risk ≈ 0 (well below threshold — low risk)
    - s_k >> q̂  → risk ≈ 1 (well above threshold — high risk)

    Args:
        nonconformity_score:  s_k ∈ [0, 1] from compute_combined_nonconformity().
        calibration_quantile: q̂_{1-α} from CalibrationStore.
        scale:                Sigmoid steepness (default 10.0 → sharp transition).

    Returns:
        Step risk ∈ [0, 1].
    """
    if not (0.0 <= nonconformity_score <= 1.0):
        raise ValueError(f"nonconformity_score must be in [0,1], got {nonconformity_score}")
    if not (0.0 <= calibration_quantile <= 1.0):
        raise ValueError(f"calibration_quantile must be in [0,1], got {calibration_quantile}")

    exponent = -scale * (nonconformity_score - calibration_quantile)
    # Clamp exponent to avoid overflow
    exponent = max(-500.0, min(500.0, exponent))
    step_risk = 1.0 / (1.0 + math.exp(exponent))
    return float(step_risk)


def compute_trajectory_risk(step_risks: list[float]) -> float:
    """
    Compute trajectory risk using the noisy-or aggregation formula.

    R_traj = 1 - Π(1 - r_k)

    Properties:
    - Monotonically increasing with more high-risk steps.
    - A single r_k = 1.0 makes R_traj = 1.0 (any proof of hallucination
      propagates to the whole trajectory).
    - Approaches 1.0 asymptotically with many medium-risk steps.

    Args:
        step_risks: List of per-hop step risk scores ∈ [0, 1].

    Returns:
        Trajectory risk ∈ [0, 1]. Returns 0.0 for empty list.
    """
    if not step_risks:
        return 0.0

    # Validate inputs
    for i, r in enumerate(step_risks):
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"step_risks[{i}]={r} must be in [0, 1]")

    # Compute product of (1 - r_k) in log space to avoid underflow
    log_product = sum(math.log(max(1.0 - r, 1e-300)) for r in step_risks)

    # Clamp to prevent floating point issues at extremes
    NOISY_OR_CLAMP_MAX = 0.999
    trajectory_risk = 1.0 - math.exp(log_product)
    return min(NOISY_OR_CLAMP_MAX, max(0.0, trajectory_risk))
