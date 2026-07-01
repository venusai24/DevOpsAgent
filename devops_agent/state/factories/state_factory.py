"""StateFactory — creates properly initialised state instances.

Provides a single place to construct initial state for a new investigation,
so that every created state has the correct default sub-state and metadata.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from ..enums.investigation_status import InvestigationStatus
from ..models.execution_metrics import ExecutionMetrics
from ..models.execution_state import ExecutionState
from ..models.failure_history import FailureHistory
from ..models.fallback_history import FallbackHistory
from ..models.guardrail_state import GuardrailState
from ..models.investigation_state import ExplicitSymptoms, InvestigationState
from ..models.recovery_state import RecoveryState
from ..models.timeout_state import TimeoutState


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StateFactory:
    """Constructs initial state models for new investigations."""

    @staticmethod
    def new_investigation(
        investigation_id: uuid.UUID | None = None,
        cluster_id: str = "",
        explicit_symptoms: dict[str, Any] | None = None,
        global_budget_seconds: int = 1800,
    ) -> InvestigationState:
        """Create a fully initialised InvestigationState for a new investigation.

        All Phase 4 sub-states are initialised with defaults so the orchestrator
        never needs to check for None before updating them.
        """
        inv_id = investigation_id or uuid.uuid4()
        now = _utcnow()

        symptoms = None
        if explicit_symptoms:
            symptoms = ExplicitSymptoms(
                cmdb_ids=tuple(explicit_symptoms.get("cmdb_ids", [])),
                tc_values=tuple(explicit_symptoms.get("tc_values", [])),
                description=explicit_symptoms.get("description", ""),
                alert_timestamp=explicit_symptoms.get("alert_timestamp"),
                raw_alert_payload=explicit_symptoms.get("raw_alert_payload", {}),
            )

        guardrails = GuardrailState(
            investigation_id=inv_id,
            investigation_wall_clock_start=now,
        )
        recovery = RecoveryState(investigation_id=inv_id)
        timeouts = TimeoutState(
            investigation_id=inv_id,
            started_at=now,
            global_budget_seconds=global_budget_seconds,
        )
        failure_history = FailureHistory(investigation_id=inv_id)
        fallback_history = FallbackHistory(investigation_id=inv_id)
        metrics = ExecutionMetrics(investigation_id=inv_id)

        return InvestigationState(
            investigation_id=inv_id,
            cluster_id=cluster_id,
            explicit_symptoms=symptoms,
            status=InvestigationStatus.CLAIMED,
            guardrails=guardrails,
            recovery=recovery,
            timeouts=timeouts,
            failure_history=failure_history,
            fallback_history=fallback_history,
            metrics=metrics,
        )

    @staticmethod
    def new_execution_state(
        investigation_id: uuid.UUID,
        cluster_id: str,
        symptom_signature: str,
        time_range_start: datetime | None = None,
        time_range_end: datetime | None = None,
    ) -> ExecutionState:
        """Create an ExecutionState row for insertion into investigation_registry."""
        return ExecutionState(
            investigation_id=investigation_id,
            cluster_id=cluster_id,
            symptom_signature=symptom_signature,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            status=InvestigationStatus.CLAIMED,
            started_at=_utcnow(),
        )
