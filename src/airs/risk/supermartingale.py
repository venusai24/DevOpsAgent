"""
Supermartingale Tracking — Module 1.4.

Implements the e-process / supermartingale used for statistical escalation
detection. The supermartingale M_t tests whether the LLM's cumulative
reasoning quality is degrading beyond acceptable bounds.

    e_k = s_k / q̂_{1-α}                      e-value at hop k
    M_t = M_{t-1} * e_k                       running product (supermartingale)
    ESCALATE if M_t >= 1/δ                     alarm threshold

Under the null hypothesis (evidence is conforming), E[e_k] ≤ 1 and M_t
is a supermartingale — meaning it doesn't tend to drift upward on average.
When the LLM consistently produces nonconforming (hallucinated) interpretations,
M_t grows rapidly and triggers escalation.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


def compute_e_value(
    nonconformity_score: float,
    calibration_quantile: float,
) -> float:
    """
    Compute the e-value (e-process contribution) for a single hop.

        e_k = s_k / q̂_{1-α}

    Properties:
    - e_k < 1: nonconformity below threshold → conforming evidence.
    - e_k = 1: exactly at the calibration threshold.
    - e_k > 1: nonconformity above threshold → suspicious.
    - E[e_k] ≤ 1 under the null (valid evidence).

    Args:
        nonconformity_score:  s_k ∈ [0, 1].
        calibration_quantile: q̂_{1-α} from CalibrationStore (must be > 0).

    Returns:
        e-value ≥ 0. Clipped at 100 to prevent single-hop M_t explosion.
    """
    if calibration_quantile <= 0.0:
        log.warning(
            "calibration_quantile=%.4f is ≤ 0 — using minimum of 0.001",
            calibration_quantile,
        )
        calibration_quantile = 0.001

    e_value = nonconformity_score / calibration_quantile
    # Cap at 100 to prevent a single catastrophic hop from immediately
    # triggering escalation before the supermartingale can stabilise
    return min(100.0, max(0.0, e_value))


def update_supermartingale(current_M: float, e_value: float) -> float:
    """
    Update the supermartingale M_t.

        M_t = M_{t-1} * e_k

    The supermartingale is a running product of e-values. Under valid evidence,
    it stays bounded. Under repeated nonconformity, it grows toward 1/δ.

    Args:
        current_M: Current supermartingale value M_{t-1} (starts at 1.0).
        e_value:   e-value for the current hop.

    Returns:
        Updated supermartingale value M_t.
    """
    if current_M < 0:
        raise ValueError(f"current_M must be non-negative, got {current_M}")

    new_M = current_M * e_value

    # Guard against numerical overflow for extremely long investigations
    if math.isinf(new_M) or math.isnan(new_M):
        log.error(
            "Supermartingale overflow: M_{t-1}=%.4f * e_k=%.4f → clamping to 1e9",
            current_M,
            e_value,
        )
        return 1e9

    return new_M


def check_escalation(supermartingale_value: float, delta: float) -> bool:
    """
    Check whether the supermartingale has exceeded the escalation threshold.

        Escalate if M_t >= 1/δ

    By Ville's inequality, the probability that M_t ever exceeds 1/δ under
    the null hypothesis (valid evidence) is at most δ. Setting δ=0.05 means
    there is at most a 5% chance of a false escalation over the entire
    investigation trajectory.

    Args:
        supermartingale_value: Current M_t.
        delta:                 Alarm level δ (e.g., 0.05 → threshold = 20).

    Returns:
        True if escalation should be triggered.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")

    threshold = 1.0 / delta
    should_escalate = supermartingale_value >= threshold

    if should_escalate:
        log.warning(
            "Supermartingale escalation triggered: M_t=%.4f >= 1/δ=%.4f",
            supermartingale_value,
            threshold,
        )

    return should_escalate
