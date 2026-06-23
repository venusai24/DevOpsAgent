"""
Context Management Models.

Data structures for the Proactive Context Management Engine:
context scoring, budget partitions, insight tiers, and context metrics.
Mirrors the ProactiveContextManagement_UnifiedArchitecture.md §4.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, computed_field


# ─── Context Scoring ──────────────────────────────────────────────────────────

class ContextScore(BaseModel):
    """
    5-dimensional relevance score for an evidence node.

    Weights (from Context Architecture §4.1):
      relevance:         0.30 — semantic proximity to incident symptom
      recency:           0.15 — exponential decay from current time
      causal_importance: 0.25 — centrality in causal chain
      uniqueness:        0.15 — information no other evidence provides
      diagnostic_value:  0.15 — impact on confidence score

    composite = weighted sum of all five dimensions.
    """
    relevance: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    causal_importance: float = Field(ge=0.0, le=1.0)
    uniqueness: float = Field(ge=0.0, le=1.0)
    diagnostic_value: float = Field(ge=0.0, le=1.0)

    # ClassVar so Pydantic does not treat WEIGHTS as a model field
    WEIGHTS: ClassVar[dict[str, float]] = {
        "relevance": 0.30,
        "recency": 0.15,
        "causal_importance": 0.25,
        "uniqueness": 0.15,
        "diagnostic_value": 0.15,
    }

    @computed_field  # type: ignore[misc]
    @property
    def composite(self) -> float:
        """Weighted composite score ∈ [0, 1]."""
        return (
            self.WEIGHTS["relevance"] * self.relevance
            + self.WEIGHTS["recency"] * self.recency
            + self.WEIGHTS["causal_importance"] * self.causal_importance
            + self.WEIGHTS["uniqueness"] * self.uniqueness
            + self.WEIGHTS["diagnostic_value"] * self.diagnostic_value
        )


# ─── Insight Tier ─────────────────────────────────────────────────────────────

class InsightTier(str, Enum):
    """
    Lifecycle tier for an evidence node in the context window.

    ACTIVE:     Full content in context — immediately accessible to LLM.
    SUMMARIZED: Compressed summary in context — accessible but lossy.
    ARCHIVED:   node_id only tracked — retrievable via ConANN if needed.
    DISCARDED:  Permanently removed — insufficient relevance/causal value.
    """
    ACTIVE = "ACTIVE"
    SUMMARIZED = "SUMMARIZED"
    ARCHIVED = "ARCHIVED"
    DISCARDED = "DISCARDED"


# ─── Summary Structures ───────────────────────────────────────────────────────

class InsightSummary(BaseModel):
    """Compressed summary of one or more evidence nodes moved to SUMMARIZED tier."""
    original_node_ids: list[str]
    hop_range: tuple[int, int]
    summary_text: str
    key_services: list[str] = Field(default_factory=list)
    key_evidence_types: list[str] = Field(default_factory=list)
    aggregate_confidence_delta: float = Field(ge=0.0, le=1.0)
    token_count: int = Field(ge=0)


class TelescopedSummaryNode(BaseModel):
    """
    Compressed summary of a run of middle-chain causal nodes (telescoping).
    Head and tail of the causal chain are preserved in full; middle is merged.
    """
    original_node_ids: list[str]
    hop_range: tuple[int, int]
    merged_finding: str
    key_services_involved: list[str] = Field(default_factory=list)
    key_evidence_types: list[str] = Field(default_factory=list)
    aggregate_confidence_delta: float = Field(ge=0.0, le=1.0)
    token_count: int = Field(ge=0)


# ─── Budget Partitions ────────────────────────────────────────────────────────

class ContextBudget(BaseModel):
    """
    Token budget partitions for the 80K context window.

    Percentages must sum to 1.0. Safety margin is never allocated.
    All token ceilings are derived from max_tokens and the percentages.
    """
    max_tokens: int = Field(default=80_000, ge=10_000)

    # Partition percentages
    constitution_pct: float = Field(default=0.08)
    playbook_pct: float = Field(default=0.10)
    active_evidence_pct: float = Field(default=0.50)
    summarized_evidence_pct: float = Field(default=0.12)
    tool_output_pct: float = Field(default=0.10)
    response_pct: float = Field(default=0.05)
    safety_margin_pct: float = Field(default=0.05)

    # Causal chain quota within active evidence
    causal_chain_quota: float = Field(default=0.60)

    # ── Derived token ceilings ──
    @computed_field  # type: ignore[misc]
    @property
    def constitution_tokens(self) -> int:
        return int(self.max_tokens * self.constitution_pct)

    @computed_field  # type: ignore[misc]
    @property
    def playbook_tokens(self) -> int:
        return int(self.max_tokens * self.playbook_pct)

    @computed_field  # type: ignore[misc]
    @property
    def active_evidence_tokens(self) -> int:
        return int(self.max_tokens * self.active_evidence_pct)

    @computed_field  # type: ignore[misc]
    @property
    def summarized_evidence_tokens(self) -> int:
        return int(self.max_tokens * self.summarized_evidence_pct)

    @computed_field  # type: ignore[misc]
    @property
    def tool_output_tokens(self) -> int:
        return int(self.max_tokens * self.tool_output_pct)

    @computed_field  # type: ignore[misc]
    @property
    def response_tokens(self) -> int:
        return int(self.max_tokens * self.response_pct)

    @computed_field  # type: ignore[misc]
    @property
    def causal_chain_tokens(self) -> int:
        return int(self.active_evidence_tokens * self.causal_chain_quota)


# ─── Insight Tiers Container ──────────────────────────────────────────────────

class InsightTiers(BaseModel):
    """
    The four-tier memory store for evidence nodes.

    active:          Full EvidenceNode objects — in context.
    summarized:      Compressed InsightSummary objects — in context.
    archived:        node_ids only (strings) — out of context, retrievable.
    discarded_count: Running count of permanently discarded nodes.
    """
    active: list[Any] = Field(default_factory=list)       # list[EvidenceNode]
    summarized: list[InsightSummary] = Field(default_factory=list)
    archived: list[str] = Field(default_factory=list)      # node_ids
    discarded_count: int = Field(default=0, ge=0)


# ─── Context Metrics ──────────────────────────────────────────────────────────

class ContextMetrics(BaseModel):
    """
    Runtime metrics about the context window state.
    Updated after every evaluate_context_activity call.
    """
    utilization_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_density: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Avg composite score of active evidence",
    )
    waste_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of active evidence with score < eviction threshold",
    )
    compression_ratio: float = Field(
        default=0.0,
        ge=0.0,
        description="Tokens saved by compression / original token count",
    )
    telescope_count: int = Field(default=0, ge=0)
    hypothesis_shift_count: int = Field(default=0, ge=0)
    eviction_count: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
