"""ThreadPoolExecutor stage timeout enforcer."""

import concurrent.futures
from collections.abc import Callable
from typing import Any

from ...orchestrator.budget.investigation_clock import InvestigationClock


class StageTimeoutEnforcer:
    def __init__(self, clock: InvestigationClock):
        self.clock = clock

    def execute_with_timeout(self, node_fn: Callable, state: dict[str, Any], node_name: str) -> dict[str, Any]:
        budget = self.clock.budget_for_node(node_name)
        
        # Buffer for checkpoints
        executor_timeout = max(1.0, budget - 5.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(node_fn, state)
            try:
                return future.result(timeout=executor_timeout)
            except concurrent.futures.TimeoutError:
                return {
                    "investigation_state": "TIMEOUT",
                    "current_node": node_name,
                    "guardrails": {
                        "global_timeout_triggered": self.clock.is_expired(),
                        "stage_timeout_triggered_at": "now",
                        "stage_timeout_node": node_name
                    }
                }
