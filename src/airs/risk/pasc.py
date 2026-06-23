"""
Pipeline-Aware Split Conformal (PASC) — Module 1.4.

PASC provides a coverage guarantee specifically for Tier 2/3 tool executions
(Investigative and Remediation tools). Before executing any state-mutating
tool, the agent must verify that the cumulative risk of the investigation
pipeline to this point does not violate the marginal coverage guarantee.

The key property:
    P(s_max > q̂_{1-α}) ≤ α

Where s_max is the maximum nonconformity score across all steps in the pipeline
leading to this tool execution. If this exceeds the calibration quantile,
the pipeline has demonstrated excessive uncertainty and the tool execution
must be escalated to a human operator.

Phase 1 Note:
    PASC is applied before every Tier 2 and Tier 3 tool execution.
    It is also checked as a final gate before the DIAGNOSE state is emitted.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def compute_pasc_joint_max(step_nonconformity_scores: list[float]) -> float:
    """
    Compute the joint maximum nonconformity score across all pipeline steps.

        s_max = max(s_1, s_2, ..., s_k)

    This is the PASC test statistic. It captures the worst-case nonconformity
    observed in the investigation pipeline up to this point.

    Args:
        step_nonconformity_scores: List of per-hop s_k values ∈ [0, 1].

    Returns:
        The maximum s_k observed. Returns 0.0 for an empty list.
    """
    if not step_nonconformity_scores:
        return 0.0

    for i, s in enumerate(step_nonconformity_scores):
        if not (0.0 <= s <= 1.0):
            raise ValueError(
                f"step_nonconformity_scores[{i}]={s} must be in [0, 1]"
            )

    return max(step_nonconformity_scores)


def validate_pasc_coverage(
    s_max: float,
    calibration_quantile: float,
) -> bool:
    """
    Validate PASC coverage before a Tier 2/3 tool execution.

    The coverage guarantee holds (tool execution is safe to proceed) iff:
        s_max ≤ q̂_{1-α}

    If s_max > q̂, the pipeline has exceeded acceptable uncertainty and the
    tool execution must be escalated (returned as False).

    Args:
        s_max:                Maximum nonconformity across all pipeline steps.
        calibration_quantile: q̂_{1-α} from CalibrationStore.

    Returns:
        True if coverage is met (safe to proceed).
        False if coverage is violated (must escalate).
    """
    coverage_met = s_max <= calibration_quantile

    if not coverage_met:
        log.warning(
            "PASC coverage violation: s_max=%.4f > q̂=%.4f — escalation required",
            s_max,
            calibration_quantile,
        )
    else:
        log.debug(
            "PASC coverage met: s_max=%.4f ≤ q̂=%.4f — tool execution authorised",
            s_max,
            calibration_quantile,
        )

    return coverage_met


def compute_pasc_for_tier3(
    step_nonconformity_scores: list[float],
    calibration_quantile: float,
) -> tuple[float, bool]:
    """
    Combined PASC computation for Tier 3 (Remediation) tool gate.

    Returns both the s_max statistic and the coverage verdict in one call.

    Args:
        step_nonconformity_scores: Per-hop s_k values from the investigation.
        calibration_quantile:      q̂_{1-α} for PASC coverage.

    Returns:
        (s_max, coverage_met): The test statistic and the pass/fail verdict.
    """
    s_max = compute_pasc_joint_max(step_nonconformity_scores)
    coverage_met = validate_pasc_coverage(s_max, calibration_quantile)
    return s_max, coverage_met
