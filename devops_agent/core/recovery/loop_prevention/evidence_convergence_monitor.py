"""Detects score stagnation in evidence loop."""

from typing import Literal

from ...config.guardrails_config import LoopPreventionConfig


class EvidenceConvergenceMonitor:
    def __init__(self, config: LoopPreventionConfig):
        self.config = config
        self.score_history: dict[str, list[float]] = {}

    def record_score(self, hypothesis_name: str, score: float) -> None:
        if hypothesis_name not in self.score_history:
            self.score_history[hypothesis_name] = []
        self.score_history[hypothesis_name].append(score)

    def assess(self, hypothesis_name: str, calls_used: int, budget: int) -> Literal["continue", "stagnant", "eliminate", "converged"]:
        history = self.score_history.get(hypothesis_name, [])
        if not history:
            return "continue"
            
        current_score = history[-1]
        
        # 1. Natural convergence
        if current_score > 0.75:
            return "converged"
            
        # 2. Elimination
        if calls_used >= budget / 2 and current_score < self.config.early_exit_score_floor:
            return "eliminate"
            
        # 3. Stagnation
        if len(history) >= self.config.score_stagnation_window:
            window = history[-self.config.score_stagnation_window:]
            score_delta = max(window) - min(window)
            if score_delta < self.config.score_stagnation_threshold:
                return "stagnant"
                
        return "continue"
