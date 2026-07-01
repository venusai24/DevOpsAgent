"""RecoveryState — aggregate view of recovery activity for one investigation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseState, _new_uuid, _utcnow


@dataclass(frozen=True)
class RecoveryAction(BaseState):
    """One recovery action taken by the RetryCoordinator or special strategies."""

    action_id: uuid.UUID = field(default_factory=_new_uuid)
    node_name: str = ""
    # "correctable_tool_retry" | "schema_repair" | "json_repair"
    # | "baseline_ref_recovery" | "hitl_escalation" | "circuit_open_fast_fail"
    action_type: str = ""
    # The failure that triggered this action.
    trigger_failure_code: str | None = None
    trigger_failure_type: str | None = None
    attempted_at: datetime = field(default_factory=_utcnow)
    succeeded: bool = False
    # Free-text outcome note (no business logic — for audit only).
    note: str | None = None


@dataclass(frozen=True)
class RecoveryState(BaseState):
    """Accumulates all recovery actions for one investigation.

    Stored alongside GuardrailState in the LangGraph checkpoint.
    """

    investigation_id: uuid.UUID = field(default_factory=_new_uuid)
    recovery_actions: tuple[RecoveryAction, ...] = ()

    # True when _handle_uncorrectable() was called and HITL escalation resulted.
    hitl_escalated_for_failure: bool = False
    hitl_escalation_node: str | None = None
    hitl_escalation_reason: str | None = None

    # True when BaselineRefRecoveryStrategy resolved INVALID_BASELINE_REF
    # without HITL.
    baseline_ref_self_healed: bool = False
    baseline_ref_heal_strategy: str | None = None  # "cache_hit" | "recomputed"

    def append_action(self, action: RecoveryAction) -> RecoveryState:
        return self.evolve(recovery_actions=self.recovery_actions + (action,))

    def mark_hitl_escalated(self, node: str, reason: str) -> RecoveryState:
        return self.evolve(
            hitl_escalated_for_failure=True,
            hitl_escalation_node=node,
            hitl_escalation_reason=reason,
        )
