"""Hard ceilings for tool execution within an agent."""

from ...config.guardrails_config import LoopPreventionConfig


class ToolCallBudget:
    def __init__(self, config: LoopPreventionConfig):
        self.config = config
        self.hypothesis_calls: dict[str, int] = {}
        self.total_evidence_calls = 0

    def record_call(self, hypothesis_name: str, tool_name: str) -> None:
        self.hypothesis_calls[hypothesis_name] = self.hypothesis_calls.get(hypothesis_name, 0) + 1
        self.total_evidence_calls += 1

    def is_hypothesis_exhausted(self, hypothesis_name: str) -> bool:
        return self.hypothesis_calls.get(hypothesis_name, 0) >= self.config.max_tool_calls_per_hypothesis

    def is_node_exhausted(self) -> bool:
        return self.total_evidence_calls >= self.config.max_tool_calls_evidence_node

    def remaining_for_hypothesis(self, hypothesis_name: str) -> int:
        return max(0, self.config.max_tool_calls_per_hypothesis - self.hypothesis_calls.get(hypothesis_name, 0))
