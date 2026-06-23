"""
Hypothesis Models.

Data structures for the hypothesis lifecycle management system.
The agent tracks up to N concurrent hypotheses about root cause,
promoting, abandoning, and confirming them as evidence accumulates.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class HypothesisStatus(str, Enum):
    """
    Lifecycle state of a hypothesis.

    ACTIVE:    Under active consideration — evidence being gathered.
    LEADING:   Currently the most probable hypothesis (only one at a time).
    ABANDONED: Evidence contradicts or makes this hypothesis implausible.
    CONFIRMED: Evidence confirms this as the root cause — investigation ends.
    """
    ACTIVE = "ACTIVE"
    LEADING = "LEADING"
    ABANDONED = "ABANDONED"
    CONFIRMED = "CONFIRMED"


class Hypothesis(BaseModel):
    """
    A candidate root cause hypothesis tracked by the HypothesisTracker.

    hypothesis_id:          Unique identifier.
    statement:              Natural language statement of the hypothesis.
    confidence:             Current confidence ∈ [0, 1].
    confidence_history:     Time series of confidence updates.
    supporting_evidence:    node_ids of evidence that supports this hypothesis.
    contradicting_evidence: node_ids of evidence that contradicts it.
    created_at_hop:         Which hop generated this hypothesis.
    last_updated_hop:       Last hop that changed this hypothesis.
    status:                 Current lifecycle state.
    """
    hypothesis_id: str
    statement: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_history: list[tuple[int, float]] = Field(
        default_factory=list,
        description="List of (hop_index, confidence) tuples",
    )
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    created_at_hop: int = Field(ge=0)
    last_updated_hop: int = Field(ge=0)
    status: HypothesisStatus = HypothesisStatus.ACTIVE

    def record_confidence(self, hop_index: int, new_confidence: float) -> None:
        """Append a confidence update to the history."""
        self.confidence = new_confidence
        self.last_updated_hop = hop_index
        self.confidence_history.append((hop_index, new_confidence))


class TelescopedSummaryNode(BaseModel):
    """
    Compressed summary of a middle segment of the causal chain (telescoping).
    Head and tail nodes are preserved in full; this represents the merged middle.
    """
    original_node_ids: list[str]
    hop_range: tuple[int, int]
    merged_finding: str
    key_services_involved: list[str] = Field(default_factory=list)
    key_evidence_types: list[str] = Field(default_factory=list)
    aggregate_confidence_delta: float = Field(ge=0.0, le=1.0)
    token_count: int = Field(ge=0)
