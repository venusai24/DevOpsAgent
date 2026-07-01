"""Wall-clock budget tracker."""

import time

from ...config.guardrails_config import TimeoutConfig


class InvestigationClock:
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.total_elapsed: float = 0.0
        self.last_start: float = 0.0
        self.paused = True

    def start(self) -> None:
        if self.paused:
            self.last_start = time.time()
            self.paused = False

    def pause(self) -> None:
        if not self.paused:
            self.total_elapsed += time.time() - self.last_start
            self.paused = True

    def resume(self) -> None:
        self.start()

    def elapsed_seconds(self) -> float:
        if self.paused:
            return self.total_elapsed
        return self.total_elapsed + (time.time() - self.last_start)

    def remaining_seconds(self) -> float:
        return max(0, self.config.global_timeout_seconds - self.elapsed_seconds())

    def is_expired(self) -> bool:
        return self.remaining_seconds() <= 0

    def budget_for_node(self, node_name: str) -> float:
        remaining = self.remaining_seconds()
        
        node_limit = {
            "context_assembly": self.config.context_assembly_timeout_s,
            "triage": self.config.triage_timeout_s,
            "evidence_collection": self.config.evidence_collection_timeout_s,
            "causal_reasoning": self.config.causal_reasoning_timeout_s,
            "report_delivery": self.config.report_delivery_timeout_s
        }.get(node_name, 300)
        
        return min(remaining, node_limit)
