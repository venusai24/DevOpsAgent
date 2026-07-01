"""FallbackHistory — log of all algorithmic fallback activations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class FallbackActivation(BaseState):
    """One algorithmic fallback activation (Domain 3 Stage 5.1b)."""

    activation_id: uuid.UUID = field(default_factory=_new_uuid)
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # "trace_degradation" | "metric_fallback" | "forced_exit_no_survivors"
    fallback_type: str = "trace_degradation"

    # Algorithmic criteria that triggered this activation.
    trigger_criteria: tuple[str, ...] = ()

    # TraceQualityAssessor raw score at trigger time.
    trace_quality_score_at_trigger: float | None = None
    trace_coverage_pct_at_trigger: float | None = None

    # The alternative path taken.
    alternative_stage: str = "5.1b"
    alternative_path_label: str = ""

    activated_at: datetime = field(default_factory=_utcnow)

    # Result of the fallback path.
    # "completed" | "partial" | "exhausted"
    fallback_outcome: str = "completed"
    fallback_confidence_degradation: str | None = None  # e.g., "HIGH → MEDIUM"

    # Key metrics from the alternative path execution.
    fallback_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FallbackHistory(BaseState):
    """Ordered log of all fallback activations for one investigation."""

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)
    activations: tuple[FallbackActivation, ...] = ()
    trace_degradation_count: int = 0
    forced_exit_count: int = 0

    def append(self, activation: FallbackActivation) -> FallbackHistory:
        td_delta = 1 if activation.fallback_type == "trace_degradation" else 0
        fe_delta = 1 if activation.fallback_type == "forced_exit_no_survivors" else 0
        return self.evolve(
            activations=self.activations + (activation,),
            trace_degradation_count=self.trace_degradation_count + td_delta,
            forced_exit_count=self.forced_exit_count + fe_delta,
        )
