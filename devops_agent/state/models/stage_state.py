"""StageState — fine-grained tracking of one ADR-001 reasoning stage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums.stage_outcome import StageOutcome
from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class StageState(BaseState):
    """Captures the execution record for a single investigation stage (0–8).

    One StageState is created when the stage begins and updated when it ends.
    Multiple StageStates per investigation exist (one per stage).  They form
    the granular audit trail inside the checkpoint chain.
    """

    stage_id: uuid.UUID = field(default_factory=_new_uuid)
    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # ADR-001 stage number (e.g., 0, 1, 2, …, 8) or a string label for
    # sub-stages (e.g., "5.1b" for Graceful Trace Degradation).
    stage_number: str = "0"

    # Human-readable label matching ADR-001 section names.
    stage_label: str = ""

    # The LangGraph node that owns this stage.
    owning_node: str = ""

    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None

    # Elapsed wall-clock seconds (populated by the persistence layer on completion).
    elapsed_seconds: float | None = None

    outcome: StageOutcome | None = None

    # Lightweight output summary stored here; heavy artifacts go to the
    # LangGraph checkpoint or the Stage 0 Artifact Cache.
    output_summary: dict[str, Any] = field(default_factory=dict)

    # Any warnings or gaps the stage produced (feeds investigation_gaps).
    stage_gaps: tuple[str, ...] = ()

    # Token spend during this stage (for §7.2 budget tracking).
    token_spend: int = 0
    tool_calls_made: int = 0

    def complete(self, outcome: StageOutcome, now: datetime | None = None,
                 output_summary: dict | None = None,
                 gaps: tuple[str, ...] = ()) -> StageState:
        """Return a closed copy of this stage."""
        ts = now or _utcnow()
        elapsed = (ts - self.started_at).total_seconds()
        return self.evolve(
            ended_at=ts,
            elapsed_seconds=elapsed,
            outcome=outcome,
            output_summary=output_summary or self.output_summary,
            stage_gaps=self.stage_gaps + gaps,
        )
