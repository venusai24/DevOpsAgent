"""WorkflowState — tracks the sequencing of agents within one investigation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseState, _new_uuid


@dataclass(frozen=True)
class WorkflowState(BaseState):
    """Records which agents have run and what the current routing decision is.

    Distinct from ExecutionState (which is the control-plane row) and from
    InvestigationState (which is the full LangGraph-managed payload).
    WorkflowState is an intermediate, lightweight representation used by the
    managers layer to reason about sequencing without deserializing the full
    checkpoint blob.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)

    # Ordered list of node names that have completed successfully.
    completed_nodes: tuple[str, ...] = ()

    # The node currently executing or about to execute.
    active_node: str | None = None

    # The node the graph will route to next (may differ from active_node
    # during HITL pause — active_node is the interrupted node, next_node
    # is where resumption will ultimately route).
    next_node: str | None = None

    # True when the graph is paused awaiting a human response.
    hitl_paused: bool = False

    # The interrupt trigger that caused the pause (if hitl_paused).
    hitl_trigger: str | None = None

    # Wall-clock when the HITL pause started (to feed the watchdog).
    hitl_paused_at: datetime | None = None

    # Accumulated wall-clock seconds spent in HITL pauses (excluded from budget).
    total_hitl_pause_seconds: float = 0.0

    # Routing decision that determined the path to the current node.
    last_routing_decision: str | None = None

    def mark_node_completed(self, node_name: str) -> WorkflowState:
        return self.evolve(
            completed_nodes=self.completed_nodes + (node_name,),
            active_node=None,
        )

    def enter_hitl(self, trigger: str, at: datetime) -> WorkflowState:
        return self.evolve(hitl_paused=True, hitl_trigger=trigger, hitl_paused_at=at)

    def exit_hitl(self, at: datetime) -> WorkflowState:
        pause_seconds = (
            (at - self.hitl_paused_at).total_seconds()
            if self.hitl_paused_at else 0.0
        )
        return self.evolve(
            hitl_paused=False,
            hitl_trigger=None,
            hitl_paused_at=None,
            total_hitl_pause_seconds=self.total_hitl_pause_seconds + pause_seconds,
        )
