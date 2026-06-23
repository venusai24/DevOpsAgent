"""
Nonconformity Score Computation — Module 1.4.

Implements the combined nonconformity score:

    s_k = w1 * LI_score + w2 * (1 - BERT_F1)

Where:
  - LI_score:  Layer-wise Information score measuring LLM epistemic uncertainty.
               Phase 1 uses a token-entropy proxy from logprobs (Groq API).
               Phase 3 will swap in true LI from hidden states (local model only).
  - BERT_F1:   BERTScore F1 between the source evidence and the LLM interpretation.
               Measures semantic fidelity — high F1 means the interpretation is
               grounded in the actual evidence.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger(__name__)


# ─── Token-entropy LI Proxy (Phase 1) ────────────────────────────────────────

def compute_token_entropy_proxy(logprobs: list[float]) -> float:
    """
    Compute a token-entropy proxy for the Layer-wise Information (LI) score.

    LI score measures the epistemic uncertainty of the LLM's interpretation.
    The true LI requires access to model hidden states (only available for local
    models). For Phase 1, we use the Shannon entropy of the output token
    distribution as a proxy, derived from Groq API logprobs.

    High entropy → model is uncertain → high LI → high nonconformity.

    Args:
        logprobs: List of log-probabilities of the chosen tokens from the
                  LLM's output (available via Groq's logprobs API option).
                  Values should be ≤ 0 (log-probability space).

    Returns:
        Normalised entropy ∈ [0, 1]. 0 = perfectly confident, 1 = maximally uncertain.
    """
    if not logprobs:
        log.warning("Empty logprobs — returning maximum uncertainty (LI=1.0)")
        return 1.0

    # Filter out any invalid values (logprobs > 0 would be invalid)
    valid_logprobs = [lp for lp in logprobs if lp <= 0.0]
    if not valid_logprobs:
        return 1.0

    # Compute per-token entropy: H_i = -p_i * log(p_i) where p_i = exp(logprob_i)
    per_token_entropies = []
    for lp in valid_logprobs:
        p = math.exp(lp)
        if p > 0:
            per_token_entropies.append(-p * lp)  # -p * log(p) = H contribution

    if not per_token_entropies:
        return 1.0

    # Mean entropy per token
    mean_entropy = sum(per_token_entropies) / len(per_token_entropies)

    # Normalise by maximum possible entropy: H_max = log(vocab_size)
    # Groq models typically have vocab_size ≈ 128K tokens → H_max ≈ 11.76
    # We use a conservative H_max = ln(128000) ≈ 11.76
    H_max = math.log(128_000)

    return min(1.0, mean_entropy / H_max)


# ─── BERTScore F1 ─────────────────────────────────────────────────────────────

def compute_bert_score(
    reference: str,
    hypothesis: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
) -> float:
    """
    Compute BERTScore F1 between reference (source evidence) and hypothesis
    (LLM interpretation of the evidence).

    High F1 (> 0.85): LLM interpretation is semantically faithful to source.
    Low F1 (< 0.30): LLM interpretation diverges substantially → unreliable.

    Args:
        reference:  The pre-filtered source evidence text (ground truth).
        hypothesis: The LLM's natural language interpretation of that evidence.
        model_type: Underlying model for BERTScore computation.
        lang:       Language code.

    Returns:
        BERTScore F1 ∈ [0, 1].
    """
    try:
        from bert_score import score as bert_score_fn

        # bert_score returns (P, R, F1) tensors
        _P, _R, F = bert_score_fn(
            cands=[hypothesis],
            refs=[reference],
            model_type=model_type,
            lang=lang,
            verbose=False,
        )
        f1_value = float(F[0].item())
        return max(0.0, min(1.0, f1_value))

    except ImportError:
        log.error(
            "bert-score library not available. "
            "Run: pip install bert-score. "
            "Returning conservative F1=0.5."
        )
        return 0.5

    except Exception as exc:
        log.warning(
            "BERTScore computation failed (%s) — returning conservative F1=0.5",
            exc,
        )
        return 0.5


# ─── Combined Nonconformity Score ────────────────────────────────────────────

def compute_combined_nonconformity(
    li_score: float,
    bert_f1_score: float,
    w1: float = 0.40,
    w2: float = 0.60,
) -> float:
    """
    Compute the combined nonconformity score for a single investigation hop.

        s_k = w1 * LI_score + w2 * (1 - BERT_F1)

    Properties:
    - s_k → 0 when LI is low (confident) AND BERT_F1 is high (grounded): ideal evidence.
    - s_k → 1 when LI is high (uncertain) AND BERT_F1 is low (hallucinated): prune.
    - Weights default to w1=0.40, w2=0.60 (BERT_F1 trusted slightly more in Phase 1
      since true LI is only a proxy).

    Args:
        li_score:      Layer-wise information score (or entropy proxy) ∈ [0, 1].
        bert_f1_score: BERTScore F1 ∈ [0, 1].
        w1:            Weight for LI component.
        w2:            Weight for (1 - BERT_F1) component.

    Returns:
        Combined nonconformity score s_k ∈ [0, 1].
    """
    if abs(w1 + w2 - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got w1={w1}, w2={w2}")

    s_k = w1 * li_score + w2 * (1.0 - bert_f1_score)
    return max(0.0, min(1.0, s_k))
