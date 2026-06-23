"""
Evidence Models.

Defines the core data structures for evidence nodes, entity descriptors,
uncertainty metrics, provenance records, and temporal bounds.
These mirror the Protobuf schema from the Design Document (Section 5.1),
implemented as Pydantic v2 models for Phase 1 JSON serialization.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer


# ─── Entity Types ─────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    """Classification of observed entities in the investigation graph."""
    HOST = "HOST"
    PROCESS = "PROCESS"
    NETWORK_FLOW = "NETWORK_FLOW"
    IDENTITY = "IDENTITY"
    SERVICE = "SERVICE"
    POD = "POD"
    DEPLOYMENT = "DEPLOYMENT"
    NAMESPACE = "NAMESPACE"
    NODE = "NODE"
    DATABASE = "DATABASE"
    CACHE = "CACHE"
    QUEUE = "QUEUE"
    STORAGE = "STORAGE"
    CONFIG = "CONFIG"
    UNKNOWN = "UNKNOWN"


# ─── Evidence Signal Types ────────────────────────────────────────────────────

class SignalSource(str, Enum):
    """The observability signal type that produced this evidence."""
    METRICS = "METRICS"
    LOGS = "LOGS"
    TRACES = "TRACES"
    K8S_STATE = "K8S_STATE"
    CODE = "CODE"
    ALERT = "ALERT"
    # DuckDB-processed CSV telemetry files exported by the GUI module.
    CSV = "CSV"



# ─── Sub-models ───────────────────────────────────────────────────────────────

class UncertaintyMetrics(BaseModel):
    """
    Quantified epistemic uncertainty for an evidence node.

    nonconformity_score: Combined s_k = w1*LI + w2*(1-BERT_F1).
    calibration_quantile: q_hat_{1-alpha} from ConANN calibration store.
    li_score: Layer-wise information score (token-entropy proxy in Phase 1).
    bert_f1_score: BERTScore F1 between source evidence and LLM interpretation.
    """
    nonconformity_score: float = Field(ge=0.0, le=1.0)
    calibration_quantile: float = Field(ge=0.0, le=1.0)
    li_score: float = Field(ge=0.0, le=1.0, description="Phase 1: token-entropy proxy")
    bert_f1_score: float = Field(ge=0.0, le=1.0)


class ProvenanceRecord(BaseModel):
    """Immutable audit trail linking evidence to the tool call that produced it."""
    mcp_server_id: str
    tool_invoked: str
    query_parameters_hash: str  # SHA-256 of serialised query params
    raw_response_digest: str    # SHA-256 of raw MCP response (for replay)
    idempotency_key: str        # Redis idempotency key


class TemporalityBound(BaseModel):
    """
    Temporal validity window for an evidence node.

    observed_at: Wall-clock time the tool response was received.
    valid_from:  Earliest timestamp the evidence describes.
    valid_until: Latest timestamp the evidence describes (None = open-ended).
    """
    observed_at: datetime
    valid_from: datetime
    valid_until: Optional[datetime] = None

    @field_serializer("observed_at", "valid_from", "valid_until")
    def serialize_datetime(self, v: Optional[datetime]) -> Optional[str]:
        return v.isoformat() if v is not None else None


class EntityDescriptor(BaseModel):
    """
    Resolved canonical entity reference.

    canonical_guid: Stable UUID produced by the EntityResolutionEngine.
    entity_type:    Classification of the entity.
    observed_identifiers: All raw identifiers observed (aliases) that were
                    unified into this canonical entity.
    """
    canonical_guid: UUID
    entity_type: EntityType
    observed_identifiers: list[str] = Field(default_factory=list)

    @field_serializer("canonical_guid")
    def serialize_uuid(self, v: UUID) -> str:
        return str(v)


# ─── Core Evidence Node ───────────────────────────────────────────────────────

class EvidenceNode(BaseModel):
    """
    An atomic unit of evidence in the Investigation Graph.

    node_id:       Stable node identifier (UUID string).
    entity:        The resolved canonical entity this evidence describes.
    signal_source: Which observability signal produced this node.
    context:       Human-readable structured description of the finding.
    uncertainty:   Quantified epistemic quality of this evidence.
    provenance:    Audit trail back to the MCP tool call.
    temporality:   Time window the evidence covers.
    hop_index:     Which investigation hop produced this node.
    is_causal:     Whether this node is on the confirmed causal chain.
    """
    node_id: str
    entity: EntityDescriptor
    signal_source: SignalSource
    context: dict[str, Any] = Field(
        description="Structured finding — keys vary by signal type",
    )
    uncertainty: UncertaintyMetrics
    provenance: ProvenanceRecord
    temporality: TemporalityBound
    hop_index: int = Field(ge=0)
    is_causal: bool = False
    context_score: Optional[float] = Field(
        default=None,
        description="Composite context relevance score (set by ContextScorer)",
        ge=0.0,
        le=1.0,
    )


# ─── Evidence Candidate (pre-node, from SignalAdapter) ───────────────────────

class EvidenceCandidate(BaseModel):
    """
    Intermediate data structure produced by a SignalAdapter after pre-filtering.
    Passed to ContextManager for scoring and budget evaluation before being
    promoted to a full EvidenceNode if it passes the relevance threshold.
    """
    entity_type: EntityType
    raw_identifiers: list[str]
    signal_source: SignalSource
    filtered_content: dict[str, Any]    # Pre-filtered, normalised signal content
    mcp_server_id: str
    tool_invoked: str
    query_parameters_hash: str
    raw_response_digest: str
    idempotency_key: str
    observed_at: datetime
    valid_from: datetime
    valid_until: Optional[datetime] = None
    hop_index: int
    token_count: int = Field(ge=0, description="Estimated token cost of this candidate")
