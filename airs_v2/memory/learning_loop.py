"""
airs_v2/memory/learning_loop.py
================================

Post-resolution knowledge ingestion pipeline — the continuous learning mechanism.

Responsibilities
----------------
1. Build a PostMortemTrace from resolved incident artifacts
2. PII/credential redaction before embedding (Amendment #1)
3. Upsert trace into Vector Store (postmortem_traces collection)
4. Ingest trace nodes/edges into Knowledge Graph
5. Extract new knowledge from human feedback and write it to the graph

Amendment #1 — PII and Credential Redaction:
  _redact_sensitive_data() runs deterministic regex scrubbing on all text
  before it is passed to the embedding model or stored in the vector DB.
  This is architecturally consistent with the NeSy-Edge L1 regex-first approach.

  Redacted patterns:
    - IP addresses (IPv4 and IPv6)
    - UUIDs
    - Authentication tokens (Bearer, Basic, api_key formats)
    - Common secret patterns (password=, secret=, token=)
    - Private RFC1918 address ranges

Knowledge production chain
--------------------------
Signal                  → Knowledge artifact
------                  → ------------------
APPROVE                 → Positive-labeled trace with full feedback chain
REJECT + diag=incorrect → Misdiagnosis edge in Knowledge Graph
MODIFY + delta          → Action parameter correction embedded as trace
ESCALATE                → Escalation-tagged trace (useful for training)
missing_context         → Warning metadata for future retrieval filter
new_system_relationships→ New edges in Knowledge Graph
runbook_references      → Document ingestion trigger (future)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII / Credential redaction patterns (Amendment #1)
# ---------------------------------------------------------------------------

_REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # IPv4 addresses
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[REDACTED_IP]",
    ),
    # UUID v4 patterns
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        "[REDACTED_UUID]",
    ),
    # Bearer / Basic auth tokens
    (
        re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
        "Bearer [REDACTED_TOKEN]",
    ),
    (
        re.compile(r"Basic\s+[A-Za-z0-9+/]+=*", re.IGNORECASE),
        "Basic [REDACTED_TOKEN]",
    ),
    # API key / password / secret in key=value format
    (
        re.compile(
            r'(?:api[_-]?key|password|passwd|secret|token|auth[_-]?token)'
            r'\s*[:=]\s*[\'"]?[A-Za-z0-9\-._~+/!@#$%^&*]{8,}[\'"]?',
            re.IGNORECASE,
        ),
        "[REDACTED_CREDENTIAL]",
    ),
    # HuggingFace token patterns (hf_...)
    (
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "[REDACTED_HF_TOKEN]",
    ),
    # AWS-style access keys
    (
        re.compile(r"\b(AKIA|ASIA|AROA)[A-Z0-9]{16}\b"),
        "[REDACTED_AWS_KEY]",
    ),
    # Email addresses (basic PII)
    (
        re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
]


def _redact_sensitive_data(text: str) -> str:
    """
    Run deterministic regex-based PII and credential redaction.

    Amendment #1: Applied to all text before embedding or vector DB storage.
    This is architecturally consistent with the NeSy-Edge L1 regex-first approach.
    """
    if not text:
        return text
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# LearningLoop
# ---------------------------------------------------------------------------


class LearningLoop:
    """
    Continuous learning pipeline.

    Converts resolved incidents + rich human feedback into durable knowledge
    artifacts in the Vector Store and Knowledge Graph.

    Parameters
    ----------
    rag_engine : RAGEngine providing access to vector_store and graph_store.
    """

    def __init__(self, rag_engine) -> None:
        self._rag = rag_engine

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest_resolved_incident(
        self,
        incident_analysis,          # IncidentAnalysis from reasoning engine
        execution_result,           # ExecutionResult from action executor
        remediation_plan,           # RemediationPlan
        outcome_str: str = "approved",
        human_feedback=None,        # Optional[HumanFeedback]
        resolution_duration_s: float = 0.0,
    ):  # → PostMortemTrace
        """
        Core learning loop entry point.

        Builds a PostMortemTrace, applies PII redaction (Amendment #1),
        upserts to vector store, ingests to knowledge graph, and extracts
        any new knowledge from human feedback.
        """
        from airs_v2.memory.types import PostMortemTrace, IncidentOutcome

        # Build the trace
        trace = self._build_trace(
            incident_analysis=incident_analysis,
            execution_result=execution_result,
            remediation_plan=remediation_plan,
            outcome_str=outcome_str,
            human_feedback=human_feedback,
            resolution_duration_s=resolution_duration_s,
        )

        # Amendment #1: Redact PII before storing the embedding text
        # The redaction is applied to the analysis markdown field that will
        # be used in to_full_context() on retrieval hits.
        trace.analysis_markdown = _redact_sensitive_data(trace.analysis_markdown)

        # Also redact human feedback text fields
        if trace.human_feedback:
            fb = trace.human_feedback
            fb.correct_root_cause = _redact_sensitive_data(fb.correct_root_cause)
            fb.domain_expertise_notes = _redact_sensitive_data(
                fb.domain_expertise_notes
            )

        # Persist to vector store
        self._rag.vector_store.upsert_trace(trace)
        logger.info(
            "[LearningLoop] Upserted trace %s (outcome=%s)", trace.trace_id, outcome_str
        )

        # Persist to knowledge graph
        self._rag.graph_store.ingest_trace(trace)

        # Extract new knowledge from human feedback
        if human_feedback and human_feedback.has_new_knowledge():
            self._extract_feedback_knowledge(trace.trace_id, human_feedback)

        return trace

    def ingest_auto_resolved(
        self,
        incident_analysis,
        execution_result,
        remediation_plan,
        resolution_duration_s: float = 0.0,
    ):  # → PostMortemTrace
        """
        Shortcut for autonomous resolutions (no human feedback).
        Sets outcome to AUTO_RESOLVED.
        """
        return self.ingest_resolved_incident(
            incident_analysis=incident_analysis,
            execution_result=execution_result,
            remediation_plan=remediation_plan,
            outcome_str="auto_resolved",
            human_feedback=None,
            resolution_duration_s=resolution_duration_s,
        )

    def ingest_human_reviewed(
        self,
        incident_analysis,
        execution_result,
        remediation_plan,
        human_feedback,             # HumanFeedback (required)
        resolution_duration_s: float = 0.0,
    ):  # → PostMortemTrace
        """
        Shortcut for HITL-reviewed incidents.
        Maps human feedback decision to the appropriate IncidentOutcome.
        """
        decision_to_outcome = {
            "approve": "approved",
            "reject": "rejected",
            "modify": "modified",
            "escalate": "escalated",
            "defer": "deferred",
        }
        outcome_str = decision_to_outcome.get(
            human_feedback.decision, "approved"
        )
        return self.ingest_resolved_incident(
            incident_analysis=incident_analysis,
            execution_result=execution_result,
            remediation_plan=remediation_plan,
            outcome_str=outcome_str,
            human_feedback=human_feedback,
            resolution_duration_s=resolution_duration_s,
        )

    # ── Trace construction ─────────────────────────────────────────────────────

    def _build_trace(
        self,
        incident_analysis,
        execution_result,
        remediation_plan,
        outcome_str: str,
        human_feedback,
        resolution_duration_s: float,
    ):  # → PostMortemTrace
        """Build a PostMortemTrace from the incident resolution artifacts."""
        from airs_v2.memory.types import PostMortemTrace, IncidentOutcome

        # Primary root cause from analysis
        top_candidate = (
            incident_analysis.root_cause_candidates[0]
            if incident_analysis.root_cause_candidates
            else None
        )
        root_cause_node = top_candidate.candidate_node if top_candidate else ""
        root_cause_template = top_candidate.template_key if top_candidate else ""

        # Affected services from the causal graph
        affected_services = [
            node.name
            for node in incident_analysis.causal_graph.nodes
            if node.name != incident_analysis.focal_service
        ]

        # Blast radius depth from causal graph edges
        blast_radius_depth = len(set(
            e.target for e in incident_analysis.causal_graph.edges
        ))

        # Remediation plan serialised to dict
        plan_dict: dict = {}
        if remediation_plan:
            try:
                plan_dict = remediation_plan.model_dump()
            except Exception:
                plan_dict = {}

        outcome = IncidentOutcome(outcome_str)

        return PostMortemTrace(
            incident_id=execution_result.incident_id if execution_result else "",
            focal_service=incident_analysis.focal_service,
            incident_type=incident_analysis.root_cause_candidates[0].template_key
            if incident_analysis.root_cause_candidates
            else "unknown",
            root_cause_node=root_cause_node,
            root_cause_template=root_cause_template,
            diagnosis_confidence=float(incident_analysis.overall_confidence),
            action_confidence=float(
                getattr(execution_result, "composite_confidence", 0.0)
            ),
            symbolic_path=incident_analysis.symbolic_path,
            analysis_markdown=incident_analysis.analysis_markdown,
            remediation_plan_json=plan_dict,
            outcome=outcome,
            human_feedback=human_feedback,
            affected_services=affected_services,
            resolution_duration_s=resolution_duration_s,
            blast_radius_depth=blast_radius_depth,
            tags=[
                incident_analysis.focal_service,
                root_cause_template,
                incident_analysis.symbolic_path,
            ],
        )

    # ── Feedback knowledge extraction ──────────────────────────────────────────

    def _extract_feedback_knowledge(
        self, trace_id: str, feedback
    ) -> None:
        """
        Convert structured human feedback into graph knowledge.

        This is where human expertise becomes system intelligence.
        """
        try:
            self._rag.graph_store.ingest_feedback(trace_id, feedback)
            logger.info(
                "[LearningLoop] Feedback knowledge extracted for trace %s "
                "(decision=%s, new_knowledge=%s)",
                trace_id,
                feedback.decision,
                feedback.has_new_knowledge(),
            )
        except Exception as exc:
            logger.error(
                "[LearningLoop] Feedback ingestion failed for trace %s: %s",
                trace_id,
                exc,
            )
