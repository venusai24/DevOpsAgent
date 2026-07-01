"""Computes best available outcome on forced loop exit."""

from typing import Literal


class ForcedExitRouter:
    def compute_exit_outcome(
        self, surviving: list[str], scores: dict[str, float], state: dict
    ) -> tuple[Literal["CONVERGED_EARLY", "AMBIGUOUS_FORCED_EXIT", "SINGLE_SURVIVOR_LOW_CONFIDENCE", "NO_SURVIVORS"], dict]:
        
        if not surviving:
            return "NO_SURVIVORS", {
                "investigation_state": "AGENT_FAILURE",
                "guardrails": {"forced_exit_triggered": True, "forced_exit_reason": "No surviving hypotheses."}
            }
            
        # Sort surviving by score descending
        ranked = sorted(surviving, key=lambda h: scores.get(h, 0.0), reverse=True)
        top_score = scores.get(ranked[0], 0.0)
        
        if top_score > 0.5:
            # Confident enough to proceed, but note it was forced
            gaps = state.get("investigation_gaps", [])
            gaps.append({"type": "forced_exit", "reason": "Budget exhausted or looping, proceeding with best."})
            return "CONVERGED_EARLY", {"investigation_gaps": gaps, "confidence_level": "MEDIUM"}
            
        if len(ranked) == 1:
            return "SINGLE_SURVIVOR_LOW_CONFIDENCE", {"confidence_level": "LOW"}
            
        # Ambiguous: top score < 0.5 and multiple candidates
        top_two_diff = top_score - scores.get(ranked[1], 0.0)
        if top_two_diff < 0.1:
            return "AMBIGUOUS_FORCED_EXIT", {
                "investigation_state": "AMBIGUOUS",
                "guardrails": {"forced_exit_triggered": True, "forced_exit_reason": "Ambiguous evidence on loop exit."}
            }
            
        # Proceed with best
        return "SINGLE_SURVIVOR_LOW_CONFIDENCE", {"confidence_level": "LOW"}
