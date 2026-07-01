from dataclasses import dataclass


@dataclass
class ConvergenceSignal:
    state_override: str | None          # "AMBIGUOUS", "INCONCLUSIVE", or None (preserve LLM decision)
    override_reason: str | None
    evidence_depth: int                 # how many tool calls were made
    hypothesis_entropy: float           # 0.0 = fully converged; 1.0 = flat prior (no info)
    top_hypothesis_gap: float           # score[0] - score[1]; large gap = clear winner
    evidence_gap_triggered: bool        # True if evidence_depth is too low to trust any score
    convergence_confidence: float       # composite 0-1 signal

class RCAConvergenceEvaluator:
    """Computes evidence-weighted stopping signal for RCA. 
    
    Principle: Trust the LLM's judgment when evidence is sufficient.
    Override only when evidence is structurally insufficient to support any conclusion.
    """
    
    MIN_EVIDENCE_DEPTH = 3           # below this, scores are not trustworthy
    CONVERGENCE_GAP_THRESHOLD = 0.25 # top score must be this much higher than next
    ENTROPY_THRESHOLD = 0.85         # above this, scores are nearly flat -> ambiguous
    
    def evaluate(
        self,
        scores: dict[str, float],
        evidence_items: dict,          # evidence_matrix from state
        tool_calls_made: int,
        llm_confidence: str,           # the LLM's own "HIGH" / "MEDIUM" / etc.
        llm_investigation_state: str,  # the LLM's own state assessment
    ) -> ConvergenceSignal:
        
        n = len(scores)
        if n == 0:
            return ConvergenceSignal(
                state_override="AMBIGUOUS",
                override_reason="No hypothesis scores available",
                evidence_depth=tool_calls_made,
                hypothesis_entropy=1.0,
                top_hypothesis_gap=0.0,
                evidence_gap_triggered=True,
                convergence_confidence=0.0
            )
        
        sorted_scores = sorted(scores.values(), reverse=True)
        top = sorted_scores[0]
        second = sorted_scores[1] if n > 1 else 0.0
        gap = top - second
        
        # Shannon entropy of the normalized score distribution
        total = sum(sorted_scores) or 1.0
        probs = [s / total for s in sorted_scores]
        import math
        entropy = -sum(p * math.log2(p + 1e-9) for p in probs) / math.log2(n + 1) if n > 0 else 0.0
        
        evidence_count = sum(len(v) for v in evidence_items.values()) if evidence_items else 0
        
        # --- Override conditions (ordered by severity) ---
        
        # 1. LLM already called AMBIGUOUS/INCONCLUSIVE -> respect it unconditionally
        if llm_investigation_state in ("AMBIGUOUS", "INCONCLUSIVE"):
            return ConvergenceSignal(
                state_override=llm_investigation_state,
                override_reason="LLM's own assessment",
                evidence_depth=evidence_count,
                hypothesis_entropy=entropy,
                top_hypothesis_gap=gap,
                evidence_gap_triggered=False,
                convergence_confidence=top
            )
        
        # 2. Evidence structurally insufficient: override to AMBIGUOUS
        # Check both tool calls made and raw evidence found
        if evidence_count < self.MIN_EVIDENCE_DEPTH and tool_calls_made < self.MIN_EVIDENCE_DEPTH:
            return ConvergenceSignal(
                state_override="AMBIGUOUS",
                override_reason=f"Insufficient evidence depth: {evidence_count} items from {tool_calls_made} tool calls",
                evidence_depth=evidence_count,
                hypothesis_entropy=entropy,
                top_hypothesis_gap=gap,
                evidence_gap_triggered=True,
                convergence_confidence=0.0
            )
        
        # 3. Scores are informationally flat (high entropy): genuinely ambiguous
        if entropy > self.ENTROPY_THRESHOLD and n > 1:
            return ConvergenceSignal(
                state_override="AMBIGUOUS",
                override_reason=f"Score entropy {entropy:.2f} exceeds threshold — hypothesis space not converging",
                evidence_depth=evidence_count,
                hypothesis_entropy=entropy,
                top_hypothesis_gap=gap,
                evidence_gap_triggered=False,
                convergence_confidence=top
            )
        
        # 4. Clear winner exists (large gap) but LLM called LOW/MEDIUM -> trust LLM
        # 5. LLM called HIGH -> trust it unconditionally (no override)
        if llm_confidence == "HIGH" or gap >= self.CONVERGENCE_GAP_THRESHOLD:
            return ConvergenceSignal(
                state_override=None,   # preserve LLM's state
                override_reason=None,
                evidence_depth=evidence_count,
                hypothesis_entropy=entropy,
                top_hypothesis_gap=gap,
                evidence_gap_triggered=False,
                convergence_confidence=top
            )
        
        # 6. Default: low gap, adequate evidence — LLM is best judge
        return ConvergenceSignal(
            state_override=None,
            override_reason=None,
            evidence_depth=evidence_count,
            hypothesis_entropy=entropy,
            top_hypothesis_gap=gap,
            evidence_gap_triggered=False,
            convergence_confidence=top
        )
