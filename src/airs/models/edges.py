"""
Edge Models.

Directed edges between EvidenceNodes in the Investigation Graph.
Encodes causal and correlational relationships discovered during investigation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EdgeType(str, Enum):
    """
    Semantic classification of a directed edge between two evidence nodes.

    CAUSED_BY:           A directly caused B (strong causal claim).
    TEMPORAL_ADJACENT:   A and B occurred within the same time window.
    MUTUALLY_IMPLIES:    The presence of A strongly implies B (and vice versa).
    MUTATED_STATE:       A was an action that mutated the state of B's entity.
    SPAWNED_PROCESS:     A's entity spawned/launched B's entity.
    CORRELATED_BY_ENTITY: A and B share a canonical entity (same service, pod, etc.)
    ESCALATES_TO:        A is evidence that warrants escalating to B's domain.
    """
    CAUSED_BY = "CAUSED_BY"
    TEMPORAL_ADJACENT = "TEMPORAL_ADJACENT"
    MUTUALLY_IMPLIES = "MUTUALLY_IMPLIES"
    MUTATED_STATE = "MUTATED_STATE"
    SPAWNED_PROCESS = "SPAWNED_PROCESS"
    CORRELATED_BY_ENTITY = "CORRELATED_BY_ENTITY"
    ESCALATES_TO = "ESCALATES_TO"


class DirectedEdge(BaseModel):
    """
    A typed, weighted, directed edge in the Investigation Graph.

    source_node_id: ID of the originating EvidenceNode.
    target_node_id: ID of the destination EvidenceNode.
    edge_type:      Semantic classification of the relationship.
    confidence:     0-1 confidence in this relationship claim.
    metadata:       Optional freeform metadata (e.g., lag_seconds, correlation_r).
    """
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
