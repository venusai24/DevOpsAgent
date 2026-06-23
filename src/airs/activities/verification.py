"""
Epistemic Verification Activity — Module 1.10.

Temporal activity: Compute nonconformity scores and update the risk state
after each tool execution. This is where the Conformal Risk Control mathematics
are applied to the new evidence.

Steps:
  1. Compute BERTScore F1 between LLM interpretation and the evidence content
  2. Compute combined nonconformity score s_k = w1*LI + w2*(1-BERT_F1)
  3. Compute step risk via sigmoid normalization
  4. Update trajectory risk (noisy-or aggregation)
  5. Compute e-value and update supermartingale M_t
  6. Check PASC coverage for Tier 2/3 tool safety gate
  7. Return updated TrajectoryRiskState

Temporal contract:
  - Pure mathematical computation — deterministic
  - No external I/O
  - Safe to retry
"""
from __future__ import annotations

import logging

from temporalio import activity

from airs.models.investigation import InvestigationState
from airs.models.risk import TrajectoryRiskState
from airs.risk.calibration import get_calibration_store
from airs.risk.nonconformity import compute_combined_nonconformity, compute_token_entropy_proxy
from airs.risk.pasc import compute_pasc_for_tier3
from airs.risk.supermartingale import check_escalation, compute_e_value, update_supermartingale
from airs.risk.trajectory_risk import compute_step_risk, compute_trajectory_risk

log = logging.getLogger(__name__)


@activity.defn(name="verify_epistemic_state")
async def verify_epistemic_state(
    state: InvestigationState,
    tool_output_logprobs: list[float],
    llm_interpretation: str,
    evidence_content: str,
    tool_tier: int = 1,
) -> tuple[InvestigationState, bool]:
    """
    Compute nonconformity scores and update TrajectoryRiskState.

    Args:
        state:                Current InvestigationState.
        tool_output_logprobs: Log-probabilities from the LLM's interpretation response.
        llm_interpretation:   LLM's text interpretation of the evidence.
        evidence_content:     Pre-filtered evidence content (ground truth).
        tool_tier:            Tier of the tool just executed (1, 2, or 3).

    Returns:
        Tuple of:
          - Updated InvestigationState (risk_state refreshed)
          - pasc_ok: bool — True if PASC coverage is satisfied (safe to proceed)
    """
    activity.logger.info(
        "Verifying epistemic state at hop %d (tool_tier=%d)",
        state.total_hop_count, tool_tier,
    )

    calibration = get_calibration_store()
    w1, w2 = calibration.get_nonconformity_weights()
    q_hat = calibration.get_q_hat("diagnostic_knowledge")
    delta = calibration.get_escalation_delta()
    lambda_threshold = calibration.get_lambda_threshold()

    # ── Step 1: Linguistic Imprecision proxy from logprobs ─────────────────────
    li_score = compute_token_entropy_proxy(tool_output_logprobs)

    # ── Step 2: BERTScore F1 between interpretation and evidence ──────────────
    bert_f1 = await _compute_bert_score_async(llm_interpretation, evidence_content)

    # ── Step 3: Combined nonconformity score ──────────────────────────────────
    nonconformity_score = compute_combined_nonconformity(
        li_score=li_score, bert_f1_score=bert_f1, w1=w1, w2=w2
    )

    # ── Step 4: Step risk (sigmoid normalization) ─────────────────────────────
    step_risk = compute_step_risk(
        nonconformity_score=nonconformity_score,
        calibration_quantile=q_hat,
    )

    # ── Step 5: Updated trajectory risk ──────────────────────────────────────
    prev_risk = state.risk_state
    all_step_risks = list(prev_risk.step_risks) + [step_risk]
    trajectory_risk = compute_trajectory_risk(all_step_risks)

    # ── Step 6: E-value and supermartingale update ────────────────────────────
    e_value = compute_e_value(
        nonconformity_score=nonconformity_score,
        calibration_quantile=q_hat,
    )
    new_M_t = update_supermartingale(
        current_M=prev_risk.supermartingale_value,
        e_value=e_value,
    )
    escalate_flag = check_escalation(new_M_t, delta=delta)

    # ── Step 7: PASC coverage for Tier 2/3 safety gate ───────────────────────
    pasc_ok = True
    if tool_tier >= 2:
        # Gather nonconformity scores from all active nodes + current
        active_scores = [
            n.uncertainty.nonconformity_score
            for n in state.insight_tiers.active
        ] + [nonconformity_score]
        _s_max, pasc_ok = compute_pasc_for_tier3(
            nonconformity_scores=active_scores,
            calibration_quantile=q_hat,
        )
        if not pasc_ok:
            activity.logger.warning(
                "PASC coverage violation at hop %d — tier %d tool blocked",
                state.total_hop_count, tool_tier,
            )

    # ── Build updated risk state ──────────────────────────────────────────────
    new_risk_state = TrajectoryRiskState(
        trajectory_risk_score=trajectory_risk,
        step_risks=all_step_risks,
        supermartingale_value=new_M_t,
        escalation_threshold=1.0 / delta,
        lambda_threshold=lambda_threshold,
        should_escalate=escalate_flag,
    )

    # Update the most recent active node's uncertainty metrics
    updated_state = _update_latest_node_uncertainty(
        state=state,
        li_score=li_score,
        bert_f1=bert_f1,
        nonconformity_score=nonconformity_score,
        q_hat=q_hat,
    )
    updated_state = updated_state.model_copy(update={"risk_state": new_risk_state})

    activity.logger.info(
        "Epistemic verification complete: "
        "s_k=%.3f, step_risk=%.3f, M_t=%.3f, escalate=%s, pasc_ok=%s",
        nonconformity_score, step_risk, new_M_t, escalate_flag, pasc_ok,
    )

    return updated_state, pasc_ok


async def _compute_bert_score_async(
    candidate: str, reference: str
) -> float:
    """
    Async wrapper for BERTScore computation.
    Returns F1 ∈ [0, 1]. Falls back to 0.5 on error.
    """
    if not candidate.strip() or not reference.strip():
        return 0.5

    try:
        import asyncio
        from bert_score import score as bert_score_fn
        # Run CPU-bound BERTScore in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        _, _, f1 = await loop.run_in_executor(
            None,
            lambda: bert_score_fn(
                [candidate], [reference],
                lang="en", verbose=False, rescale_with_baseline=True,
            ),
        )
        return float(f1[0].item())
    except Exception as e:
        log.warning("BERTScore failed: %s — using 0.5 fallback", e)
        return 0.5


def _update_latest_node_uncertainty(
    state: InvestigationState,
    li_score: float,
    bert_f1: float,
    nonconformity_score: float,
    q_hat: float,
) -> InvestigationState:
    """Update uncertainty metrics on the most recently added evidence node."""
    active = list(state.insight_tiers.active)
    if not active:
        return state

    # The latest node is last in the active list
    latest = active[-1]
    from airs.models.evidence import UncertaintyMetrics
    updated_uncertainty = UncertaintyMetrics(
        nonconformity_score=nonconformity_score,
        calibration_quantile=q_hat,
        li_score=li_score,
        bert_f1_score=bert_f1,
    )
    updated_node = latest.model_copy(update={"uncertainty": updated_uncertainty})
    active[-1] = updated_node

    new_tiers = state.insight_tiers.model_copy(update={"active": active})
    return state.model_copy(update={"insight_tiers": new_tiers})
