"""Deterministic control flow nodes."""

from typing import Any

from ..state import InvestigationState


def deduplication_node(state: InvestigationState) -> dict[str, Any]:
    """Stage -1: Deduplication query."""
    # In a real implementation, this would query the PostgresInvestigationRepository
    # For now, we assume no overlap by default unless injected
    if "dedup_decision" not in state:
        return {"dedup_decision": "NEW"}
    return {}

def save_stage0_artifacts_node(state: InvestigationState) -> dict[str, Any]:
    """Writes Stage 0 artifacts to external cache."""
    # Mocking external write
    return {"stage0_artifacts_available": True}

def report_delivery_node(state: InvestigationState) -> dict[str, Any]:
    """Serializes final report and triggers notifications."""
    return {"investigation_state": "complete"}

def duplicate_halt_node(state: InvestigationState) -> dict[str, Any]:
    """Halts execution because the investigation is a duplicate."""
    return {"investigation_state": "complete"}
