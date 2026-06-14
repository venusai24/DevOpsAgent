"""
airs_v2/memory/types.py
=======================

Pydantic v2 data models for the RAG Memory Subsystem (Stage 5).

Type hierarchy
--------------
HumanFeedback       — Multi-dimensional structured HITL feedback.
IncidentOutcome     — Enum: approved | rejected | modified | auto_resolved | escalated | deferred.
PostMortemTrace     — Complete lifecycle record of a resolved incident.
DocumentChunk       — A contextual chunk from the curated documentation corpus.
RAGQuery            — Input parameters for RAG retrieval.
RAGResult           — A single retrieved item (trace or doc chunk) with scores.
RAGResponse         — Full response from the RAG engine for one query.

Design notes
------------
Amendment #2 — Structured incident chunking:
  PostMortemTrace.retrieval_embedding_text() → ~500-token focused retrieval vector.
  PostMortemTrace.to_full_context() → Full text for LLM context expansion on hit.
  This mirrors the parent-child pattern used for documentation chunks.

Amendment #5 — Enriched metadata schemas:
  Incidents: +resolution_duration_s, +symbolic_path, +blast_radius_depth
  Docs: +source_repository, +last_validated, +deprecated
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Feedback schema
# ---------------------------------------------------------------------------


class HumanFeedback(BaseModel):
    """
    Multi-dimensional structured HITL feedback from the Action phase reviewer.

    Four feedback dimensions capture the full spectrum of operational knowledge:
      1. Decision Signal — what the human decided
      2. Diagnostic Feedback — whether the AI's root-cause analysis was correct
      3. Action Feedback — whether the proposed remediation was correct
      4. Knowledge Contribution — new organizational knowledge from the expert
    """

    # ── Dimension 1: Decision Signal ─────────────────────────────────────────
    decision: Literal["approve", "reject", "modify", "escalate", "defer"]
    reviewer_id: str = ""

    # ── Dimension 2: Diagnostic Feedback ─────────────────────────────────────
    diagnosis_accuracy: Literal["correct", "partial", "incorrect"] = "correct"
    correct_root_cause: str = ""           # If diagnosis_accuracy != "correct"
    missing_context: list[str] = Field(default_factory=list)
    causal_path_assessment: Literal["correct", "incomplete", "wrong"] = "correct"

    # ── Dimension 3: Action Feedback ─────────────────────────────────────────
    action_appropriateness: Literal["correct", "partially_correct", "wrong"] = "correct"
    parameter_accuracy: Literal["correct", "needs_adjustment", "wrong"] = "correct"
    modifications_applied: dict[str, Any] = Field(default_factory=dict)
    alternative_action: str = ""
    risk_assessment_accuracy: Literal["correct", "overestimated", "underestimated"] = "correct"

    # ── Dimension 4: Knowledge Contribution ──────────────────────────────────
    new_troubleshooting_steps: list[str] = Field(default_factory=list)
    new_system_relationships: list[str] = Field(default_factory=list)
    runbook_references: list[str] = Field(default_factory=list)
    domain_expertise_notes: str = ""
    severity_assessment: Literal["correct", "overestimated", "underestimated"] = "correct"
    confidence_calibration: Literal[
        "well_calibrated", "overconfident", "underconfident"
    ] = "well_calibrated"

    # ── Metadata ──────────────────────────────────────────────────────────────
    feedback_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    time_to_feedback_s: float = 0.0        # Latency from HITL prompt to response

    def has_new_knowledge(self) -> bool:
        """True if this feedback produced any new organisational knowledge."""
        return bool(
            self.new_troubleshooting_steps
            or self.new_system_relationships
            or self.runbook_references
            or self.domain_expertise_notes
            or (self.diagnosis_accuracy == "incorrect" and self.correct_root_cause)
            or self.modifications_applied
        )


# ---------------------------------------------------------------------------
# Incident outcome enum
# ---------------------------------------------------------------------------


class IncidentOutcome(str, enum.Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    AUTO_RESOLVED = "auto_resolved"
    ESCALATED = "escalated"
    DEFERRED = "deferred"


# ---------------------------------------------------------------------------
# Post-mortem trace
# ---------------------------------------------------------------------------


class PostMortemTrace(BaseModel):
    """
    Complete lifecycle record of a resolved incident.

    Amendment #2 — Structured Chunking:
      retrieval_embedding_text() → compact ~500-token focused vector for retrieval.
      to_full_context()          → expanded text returned to the LLM on cache hit.

    Amendment #5 — Enriched Metadata:
      resolution_duration_s, symbolic_path, blast_radius_depth added for
      richer pre-filtering and temporal decay weighting.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str = ""
    focal_service: str = ""
    incident_type: str = ""                 # Primary template_key
    root_cause_node: str = ""
    root_cause_template: str = ""
    diagnosis_confidence: float = 0.0       # From IncidentAnalysis
    action_confidence: float = 0.0          # Composite confidence
    symbolic_path: str = "SYMBOLIC_FAST"    # Amendment #5: SYMBOLIC_FAST|CBR_GUIDED|NEURAL_FULL
    analysis_markdown: str = ""
    remediation_plan_json: dict[str, Any] = Field(default_factory=dict)
    outcome: IncidentOutcome = IncidentOutcome.APPROVED
    human_feedback: Optional[HumanFeedback] = None
    affected_services: list[str] = Field(default_factory=list)
    resolution_duration_s: float = 0.0     # Amendment #5: resolution speed
    blast_radius_depth: int = 0            # Amendment #5: topology impact depth
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Amendment #2: Structured Chunking ────────────────────────────────────

    def retrieval_embedding_text(self) -> str:
        """
        Compact (~500-token) retrieval unit for vector embedding.

        Contains only the highest-signal fields needed for similarity search:
        service identity, failure mode, root cause, action taken, and outcome.
        This is what gets embedded — NOT the full analysis markdown.
        """
        feedback_summary = ""
        if self.human_feedback:
            fb = self.human_feedback
            if fb.diagnosis_accuracy != "correct" and fb.correct_root_cause:
                feedback_summary = (
                    f"Human correction: root cause was '{fb.correct_root_cause}'. "
                )
            if fb.alternative_action:
                feedback_summary += f"Recommended action: {fb.alternative_action}. "
            if fb.domain_expertise_notes:
                feedback_summary += f"Expert notes: {fb.domain_expertise_notes}"

        affected = ", ".join(self.affected_services) if self.affected_services else "none"
        action = (
            self.remediation_plan_json.get("actions", [{}])[0].get("kind", "unknown")
            if self.remediation_plan_json.get("actions")
            else "no action"
        )

        return (
            f"Incident on service '{self.focal_service}'. "
            f"Failure type: {self.incident_type}. "
            f"Root cause: {self.root_cause_node} ({self.root_cause_template}). "
            f"Also affected: {affected}. "
            f"Diagnosis confidence: {self.diagnosis_confidence:.2f}. "
            f"Action taken: {action}. "
            f"Outcome: {self.outcome.value}. "
            f"Symbolic path: {self.symbolic_path}. "
            f"{feedback_summary}"
        ).strip()

    def to_full_context(self) -> str:
        """
        Expanded context returned to the LLM after a retrieval hit.

        Contains the full analysis markdown and remediation details — this
        is the 'parent chunk' content fetched on match for LLM context expansion.
        """
        lines = [
            f"# Post-Mortem: {self.focal_service} — {self.incident_type}",
            f"**Trace ID**: {self.trace_id}",
            f"**Root Cause**: {self.root_cause_node} via `{self.root_cause_template}`",
            f"**Outcome**: {self.outcome.value} | "
            f"**Confidence**: {self.diagnosis_confidence:.0%} | "
            f"**Resolved in**: {self.resolution_duration_s:.0f}s",
            "",
            "## Diagnosis",
            self.analysis_markdown or "*No analysis markdown stored.*",
            "",
        ]
        if self.remediation_plan_json:
            lines += [
                "## Remediation Plan",
                f"```json\n{self.remediation_plan_json}\n```",
                "",
            ]
        if self.human_feedback:
            fb = self.human_feedback
            lines += [
                "## Human Feedback",
                f"- Decision: **{fb.decision}**",
                f"- Diagnosis accuracy: {fb.diagnosis_accuracy}",
                f"- Action appropriateness: {fb.action_appropriateness}",
            ]
            if fb.correct_root_cause:
                lines.append(f"- Corrected root cause: {fb.correct_root_cause}")
            if fb.domain_expertise_notes:
                lines.append(f"- Expert notes: {fb.domain_expertise_notes}")
        return "\n".join(lines)

    def to_vector_metadata(self) -> dict[str, Any]:
        """
        Deterministic metadata envelope for pre-filtering during retrieval.

        Amendment #5: includes resolution_duration_s, symbolic_path,
        blast_radius_depth in addition to base fields.
        """
        return {
            "trace_id": self.trace_id,
            "incident_id": self.incident_id,
            "focal_service": self.focal_service,
            "incident_type": self.incident_type,
            "root_cause_node": self.root_cause_node,
            "outcome": self.outcome.value,
            "diagnosis_confidence": round(self.diagnosis_confidence, 4),
            "action_confidence": round(self.action_confidence, 4),
            # Amendment #5 fields
            "symbolic_path": self.symbolic_path,
            "resolution_duration_s": round(self.resolution_duration_s, 2),
            "blast_radius_depth": self.blast_radius_depth,
            # Feedback quality signal
            "has_human_feedback": self.human_feedback is not None,
            "diagnosis_accuracy": (
                self.human_feedback.diagnosis_accuracy
                if self.human_feedback
                else "unknown"
            ),
            "reviewer_id": (
                self.human_feedback.reviewer_id if self.human_feedback else ""
            ),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Documentation chunk
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """
    A contextual chunk from the curated static document corpus.

    Uses parent-child chunking:
      - child chunks (~300 tokens) are embedded for precise retrieval
      - parent_chunk_text (~1500 tokens) is retrieved on match for full context
      - contextual_prefix (Anthropic-style) enriches the child embedding

    Amendment #5 — Enriched Metadata:
      source_repository, last_validated, deprecated added for provenance
      tracking and temporal freshness filtering.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_title: str = ""
    document_type: Literal[
        "runbook", "sop", "architecture", "troubleshooting", "dependency"
    ] = "runbook"
    section_title: str = ""
    chunk_text: str = ""                   # The child chunk (~300 tokens)
    contextual_prefix: str = ""            # Anthropic-style retrieval enrichment
    parent_chunk_text: str = ""            # Full parent section (~1500 tokens)
    parent_chunk_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    services_mentioned: list[str] = Field(default_factory=list)
    action_kinds_mentioned: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    # Amendment #5 fields
    source_repository: str = ""            # e.g. "Corpus/Diagnosis", "runbook-site"
    last_validated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    deprecated: bool = False               # Safety filter: exclude outdated procedures

    def to_embedding_text(self) -> str:
        """Combine contextual prefix + child chunk text for vector embedding."""
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.chunk_text}"
        return self.chunk_text

    def to_vector_metadata(self) -> dict[str, Any]:
        """
        Metadata envelope for pre-filtering during retrieval.

        Amendment #5: includes source_repository, last_validated, deprecated.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_type": self.document_type,
            "section_title": self.section_title,
            "parent_chunk_id": self.parent_chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "services_mentioned": ",".join(self.services_mentioned),
            "action_kinds_mentioned": ",".join(self.action_kinds_mentioned),
            "version": self.version,
            # Amendment #5 fields
            "source_repository": self.source_repository,
            "last_validated": self.last_validated,
            "deprecated": str(self.deprecated),  # ChromaDB stores as str
        }


# ---------------------------------------------------------------------------
# RAG query / result / response
# ---------------------------------------------------------------------------


class RAGQuery(BaseModel):
    """Input parameters for a single RAG retrieval operation."""

    query_text: str
    focal_service: str = ""
    incident_type: str = ""
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=50, ge=5, le=200)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    outcome_filter: Optional[IncidentOutcome] = None
    exclude_rejected: bool = True
    exclude_deprecated_docs: bool = True   # Amendment #5: safety filter
    include_documentation: bool = True


class RAGResult(BaseModel):
    """A single retrieved item from either the incidents or documentation collection."""

    trace: Optional[PostMortemTrace] = None
    document_chunk: Optional[DocumentChunk] = None
    source: Literal["incident", "documentation"] = "incident"
    similarity_score: float = 0.0
    rerank_score: Optional[float] = None   # Set after cross-encoder reranking
    retrieval_rank: int = 0

    def get_text_for_reranking(self) -> str:
        """Return the text used as the candidate document in cross-encoder reranking."""
        if self.source == "incident" and self.trace:
            return self.trace.retrieval_embedding_text()
        if self.source == "documentation" and self.document_chunk:
            return self.document_chunk.to_embedding_text()
        return ""

    def get_context_for_llm(self) -> str:
        """Return the full expanded context injected into the LLM prompt on a hit."""
        if self.source == "incident" and self.trace:
            return self.trace.to_full_context()
        if self.source == "documentation" and self.document_chunk:
            # Return parent chunk for full contextual grounding
            if self.document_chunk.parent_chunk_text:
                return (
                    f"**{self.document_chunk.document_title}"
                    f" — {self.document_chunk.section_title}**\n\n"
                    f"{self.document_chunk.parent_chunk_text}"
                )
            return (
                f"**{self.document_chunk.document_title}**\n\n"
                f"{self.document_chunk.chunk_text}"
            )
        return ""


class RAGResponse(BaseModel):
    """Full response from the RAG engine for one retrieval query."""

    query: RAGQuery
    results: list[RAGResult] = Field(default_factory=list)
    incident_results_count: int = 0
    documentation_results_count: int = 0
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    context_precision: float = 0.0        # Fraction of results with rerank_score > 0
    hierarchical_fallback_triggered: bool = False  # True when docs were queried

    def to_few_shot_context(self, max_results: int = 3) -> str:
        """
        Format top results as structured context for LLM injection.

        Respects MAX_RAG_CONTEXT_TOKENS by limiting result count.
        Returns empty string if no results (RAG system has no relevant precedents yet).
        """
        if not self.results:
            return ""

        sections: list[str] = ["## RAG Context: Retrieved Knowledge\n"]
        shown = self.results[:max_results]

        for i, result in enumerate(shown, start=1):
            source_label = (
                f"Historical Incident #{i}"
                if result.source == "incident"
                else f"Documentation #{i}"
            )
            score_label = (
                f"rerank={result.rerank_score:.3f}"
                if result.rerank_score is not None
                else f"similarity={result.similarity_score:.3f}"
            )
            sections.append(f"### [{source_label}] ({score_label})")
            sections.append(result.get_context_for_llm())
            sections.append("")

        source_summary = (
            f"*{self.incident_results_count} incident(s), "
            f"{self.documentation_results_count} documentation chunk(s) retrieved. "
            f"Fallback to docs: {self.hierarchical_fallback_triggered}.*"
        )
        sections.append(source_summary)
        return "\n".join(sections)
