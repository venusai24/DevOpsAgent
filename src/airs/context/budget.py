"""
Context Budget Tracker — Module 1.7.

Real-time token accounting across all 6 context partitions.
Tracks current utilisation and determines when pressure-based eviction
must be triggered.

Responsibilities:
  - Account for each partition's current token usage
  - Compute utilisation ratios
  - Determine if the active evidence partition is under pressure
  - Gate new evidence admission based on token headroom
"""
from __future__ import annotations

import logging
import math
from typing import Any

from airs.models.context import ContextBudget, ContextMetrics

log = logging.getLogger(__name__)


class BudgetTracker:
    """
    Real-time token accounting for the 6-partition context budget.

    Maintains a running tally of tokens consumed per partition.
    Updates are synchronous — this is a lightweight accounting object.
    """

    def __init__(self, budget: ContextBudget) -> None:
        self._budget = budget
        self._used: dict[str, int] = {
            "constitution": 0,
            "playbook": 0,
            "active_evidence": 0,
            "summarized_evidence": 0,
            "tool_output": 0,
            "response": 0,
        }

    @property
    def budget(self) -> ContextBudget:
        return self._budget

    # ─── Token counting ───────────────────────────────────────────────────────

    def count_tokens(self, text: Any) -> int:
        """
        Estimate token count for arbitrary content.

        Uses tiktoken (cl100k_base) if available.
        Falls back to len(str) / 4 approximation.

        Returns:
            Estimated token count ≥ 1.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            import json
            as_str = json.dumps(text) if not isinstance(text, str) else text
            return max(1, len(enc.encode(as_str)))
        except Exception:
            import json
            as_str = json.dumps(text) if not isinstance(text, str) else text
            return max(1, len(as_str) // 4)

    # ─── Usage accounting ─────────────────────────────────────────────────────

    def set_partition_usage(self, partition: str, tokens: int) -> None:
        """Set absolute token usage for a partition (used for constitution, playbook)."""
        if partition not in self._used:
            raise KeyError(f"Unknown partition: {partition}")
        self._used[partition] = max(0, tokens)

    def add_to_partition(self, partition: str, tokens: int) -> None:
        """Add tokens to a partition (used when adding evidence nodes)."""
        if partition not in self._used:
            raise KeyError(f"Unknown partition: {partition}")
        self._used[partition] = max(0, self._used[partition] + tokens)

    def subtract_from_partition(self, partition: str, tokens: int) -> None:
        """Remove tokens from a partition (used when evicting evidence nodes)."""
        if partition not in self._used:
            raise KeyError(f"Unknown partition: {partition}")
        self._used[partition] = max(0, self._used[partition] - tokens)

    # ─── Capacity queries ─────────────────────────────────────────────────────

    def get_headroom(self, partition: str) -> int:
        """Available token headroom in a partition."""
        ceiling = self._get_ceiling(partition)
        return max(0, ceiling - self._used.get(partition, 0))

    def get_utilisation(self, partition: str) -> float:
        """Utilisation ratio for a partition ∈ [0, 1]."""
        ceiling = self._get_ceiling(partition)
        if ceiling == 0:
            return 1.0
        return min(1.0, self._used.get(partition, 0) / ceiling)

    def can_admit(self, partition: str, tokens: int) -> bool:
        """True if there is headroom for `tokens` more tokens in this partition."""
        return self.get_headroom(partition) >= tokens

    def is_under_pressure(
        self,
        pressure_threshold: float = 0.85,
    ) -> bool:
        """
        True if the active evidence partition utilisation >= pressure_threshold.
        This triggers the eviction procedure in the ContextManager.
        """
        return self.get_utilisation("active_evidence") >= pressure_threshold

    def total_tokens_used(self) -> int:
        """Total tokens used across all partitions."""
        return sum(self._used.values())

    def total_utilisation(self) -> float:
        """Total context window utilisation ∈ [0, 1]."""
        return min(1.0, self.total_tokens_used() / self._budget.max_tokens)

    def compute_metrics(self, avg_composite_score: float = 0.0) -> ContextMetrics:
        """Produce a ContextMetrics snapshot of current state."""
        active_used = self._used.get("active_evidence", 0)
        active_ceiling = self._budget.active_evidence_tokens

        return ContextMetrics(
            utilization_ratio=self.total_utilisation(),
            relevance_density=avg_composite_score,
            waste_ratio=0.0,  # Computed by ContextManager based on scores
            compression_ratio=0.0,  # Computed by ContextManager after compression
            total_tokens_used=self.total_tokens_used(),
        )

    # ─── Private ──────────────────────────────────────────────────────────────

    def _get_ceiling(self, partition: str) -> int:
        ceilings = {
            "constitution": self._budget.constitution_tokens,
            "playbook": self._budget.playbook_tokens,
            "active_evidence": self._budget.active_evidence_tokens,
            "summarized_evidence": self._budget.summarized_evidence_tokens,
            "tool_output": self._budget.tool_output_tokens,
            "response": self._budget.response_tokens,
        }
        if partition not in ceilings:
            raise KeyError(f"Unknown partition: {partition}")
        return ceilings[partition]
