"""GuardrailWrappedEvidenceLoop (Domain 2 Integration)."""

from typing import Any

from ...core.recovery.loop_prevention.evidence_convergence_monitor import EvidenceConvergenceMonitor
from ...core.recovery.loop_prevention.forced_exit_router import ForcedExitRouter
from ...core.recovery.loop_prevention.stage_progress_tracker import StageProgressTracker
from ...core.recovery.loop_prevention.tool_call_budget import ToolCallBudget


class GuardrailWrappedEvidenceLoop:
    def __init__(self, budget: ToolCallBudget, convergence: EvidenceConvergenceMonitor, progress: StageProgressTracker, exit_router: ForcedExitRouter):
        self.budget = budget
        self.convergence = convergence
        self.progress = progress
        self.exit_router = exit_router

    def run(self, agent: Any, state: dict[str, Any]) -> dict[str, Any]:
        """
        Runs the agent's RCA reasoning loop until natural convergence or forced exit.
        """
        surviving = state.get("surviving_hypotheses", [])
        scores = state.get("updated_hypothesis_scores", {})
        
        while True:
            # 1. Global Stopping Condition (ADR-001 §5.5)
            if self._global_stopping_condition_met(scores, surviving):
                state["investigation_state"] = "active"
                return state

            # 2. Node Budget Exhaustion
            if self.budget.is_node_exhausted():
                outcome, outcome_state = self.exit_router.compute_exit_outcome(surviving, scores, state)
                state.update(outcome_state)
                return state
                
            # 3. Structural Looping (Replay/Stagnation)
            if self.progress.is_looping():
                outcome, outcome_state = self.exit_router.compute_exit_outcome(surviving, scores, state)
                state.update(outcome_state)
                return state

            # Run LLM (mocked here, in a real system this calls agent_fn)
            # 4. Agent Execution (which internally goes through Domain 1 RetryCoordinator)
            result = agent.execute_step(state) if hasattr(agent, "execute_step") else {"tool_name": "mock", "params": {}}
            
            tool_name = result.get("tool_name", "mock_tool")
            params = result.get("params", {})
            current_hypothesis = result.get("hypothesis", surviving[0] if surviving else "mock_hyp")
            
            # Record tracking
            self.budget.record_call(current_hypothesis, tool_name)
            self.progress.record_tool_call(tool_name, params)
            
            new_score = scores.get(current_hypothesis, 0.0)
            self.convergence.record_score(current_hypothesis, new_score)
            
            # Assess convergence
            assessment = self.convergence.assess(current_hypothesis, self.budget.hypothesis_calls.get(current_hypothesis, 0), self.budget.config.max_tool_calls_per_hypothesis)
            
            if assessment == "eliminate":
                if current_hypothesis in surviving:
                    surviving.remove(current_hypothesis)
                state["eliminated_hypotheses"] = state.get("eliminated_hypotheses", []) + [{"name": current_hypothesis, "reason": "Guardrail elimination"}]
            elif assessment == "stagnant" or self.budget.is_hypothesis_exhausted(current_hypothesis):
                # Budget zeroed for this hypothesis, but it remains surviving
                pass

            state["surviving_hypotheses"] = surviving
            
            # Simple break for simulation purposes if this isn't integrated fully
            break

        return state

    def _global_stopping_condition_met(self, scores: dict[str, float], surviving: list[str]) -> bool:
        if not surviving:
            return True
        top_score = max([scores.get(h, 0.0) for h in surviving] + [0.0])
        if len(surviving) == 1 and top_score > 0.75:
            return True
        if top_score > 0.5 and len(surviving) == 0: # all scored
            return True
        return False
