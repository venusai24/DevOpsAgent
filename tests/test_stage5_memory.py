"""
tests/test_stage5_memory.py
============================

Unit tests for the Stage 5 RAG Memory Subsystem.

Coverage
--------
- PostMortemTrace: retrieval_embedding_text(), to_full_context(), to_vector_metadata()
- HumanFeedback: has_new_knowledge(), all 5 feedback dimensions
- DocumentChunk: to_embedding_text(), to_vector_metadata(), Amendment #5 fields
- RAGQuery: defaults and field validation
- RAGResult: get_text_for_reranking(), get_context_for_llm()
- RAGResponse: to_few_shot_context(), empty / populated cases
- LearningLoop: PII redaction (_redact_sensitive_data)
- ConfidenceEngine: composite scoring, HITL threshold
- FeedbackCollector: CLI JSON parsing, auto-feedback creation
- NetworkXGraphStore: ingest_trace(), get_service_incident_history(), get_pattern_frequency()

All tests are pure unit tests — no network I/O, no ChromaDB, no model loading.
"""

from __future__ import annotations

import json
import math
import uuid
from unittest.mock import MagicMock, patch

import pytest

from airs_v2.action.confidence import ConfidenceEngine
from airs_v2.action.feedback import FeedbackCollector
from airs_v2.action.types import DecisionOutcome, PolicyDecision
from airs_v2.memory.graph_store import NetworkXGraphStore
from airs_v2.memory.learning_loop import _redact_sensitive_data
from airs_v2.memory.types import (
    DocumentChunk,
    HumanFeedback,
    IncidentOutcome,
    PostMortemTrace,
    RAGQuery,
    RAGResponse,
    RAGResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def approved_trace() -> PostMortemTrace:
    """A fully populated PostMortemTrace representing an approved resolution."""
    return PostMortemTrace(
        trace_id="trace-001",
        incident_id="inc-2026-001",
        focal_service="payments-service",
        incident_type="connection_pool_exhausted",
        root_cause_node="payments-db",
        root_cause_template="connection_pool_exhausted",
        diagnosis_confidence=0.88,
        action_confidence=0.72,
        symbolic_path="SYMBOLIC_FAST",
        analysis_markdown="## Diagnosis\nThe connection pool to payments-db was exhausted due to a leaked transaction.\n\n## Root Cause\nPayments-db connection pool limit (100) reached.",
        remediation_plan_json={
            "actions": [{"kind": "restart_pod", "target": "payments-db-replica-0"}]
        },
        outcome=IncidentOutcome.APPROVED,
        human_feedback=None,
        affected_services=["checkout-service", "order-service"],
        resolution_duration_s=142.5,
        blast_radius_depth=2,
        tags=["payments-service", "connection_pool_exhausted", "SYMBOLIC_FAST"],
    )


@pytest.fixture
def approved_trace_with_feedback() -> PostMortemTrace:
    """An approved trace with rich multi-dimensional human feedback."""
    feedback = HumanFeedback(
        decision="modify",
        reviewer_id="sre-alice",
        diagnosis_accuracy="partial",
        correct_root_cause="redis-cache",
        causal_path_assessment="incomplete",
        action_appropriateness="correct",
        parameter_accuracy="needs_adjustment",
        modifications_applied={"replicas": 3},
        new_troubleshooting_steps=["Check Redis eviction policy", "Verify TTL configuration"],
        new_system_relationships=["payments-service -> redis-cache"],
        runbook_references=["runbook://redis-eviction"],
        domain_expertise_notes="Redis memory pressure was the upstream trigger.",
        time_to_feedback_s=127.3,
    )
    return PostMortemTrace(
        trace_id="trace-002",
        incident_id="inc-2026-002",
        focal_service="payments-service",
        incident_type="connection_pool_exhausted",
        root_cause_node="payments-db",
        root_cause_template="connection_pool_exhausted",
        diagnosis_confidence=0.72,
        action_confidence=0.65,
        symbolic_path="CBR_GUIDED",
        analysis_markdown="## Diagnosis\nConnection pool exhausted, possible upstream cache pressure.",
        remediation_plan_json={"actions": [{"kind": "scale_deployment"}]},
        outcome=IncidentOutcome.MODIFIED,
        human_feedback=feedback,
        affected_services=["checkout-service"],
        resolution_duration_s=310.0,
        blast_radius_depth=1,
    )


@pytest.fixture
def doc_chunk() -> DocumentChunk:
    """A documentation chunk with Amendment #5 fields."""
    return DocumentChunk(
        chunk_id="chunk-001",
        document_id="redis-eviction-runbook",
        document_title="Redis Eviction Policy Runbook",
        document_type="runbook",
        section_title="Diagnosing Memory Pressure",
        chunk_text="When Redis OOM errors appear, check the maxmemory-policy setting. "
                   "Use INFO memory to inspect used_memory_peak.",
        contextual_prefix="This excerpt is from the SRE Runbook titled 'Redis Eviction Policy Runbook'. "
                          "It is located in the 'Diagnosing Memory Pressure' section.",
        parent_chunk_text="## Diagnosing Memory Pressure\n\nWhen Redis OOM errors appear...\n\n"
                          "## Remediation\nSet maxmemory-policy to allkeys-lru.",
        parent_chunk_id="parent-chunk-001",
        chunk_index=0,
        total_chunks=3,
        services_mentioned=["redis-cache"],
        action_kinds_mentioned=["restart_pod"],
        version="1.0.0",
        source_repository="Corpus/Remediation",
        deprecated=False,
    )


@pytest.fixture
def graph_store(tmp_path) -> NetworkXGraphStore:
    """A fresh NetworkXGraphStore backed by a tmp file."""
    persist_path = tmp_path / "test_graph.json"
    return NetworkXGraphStore(persist_path=str(persist_path))


# ---------------------------------------------------------------------------
# PostMortemTrace tests
# ---------------------------------------------------------------------------


class TestPostMortemTrace:

    def test_retrieval_embedding_text_is_compact(self, approved_trace):
        """Embedding text should be a focused sentence-level summary < 600 chars."""
        text = approved_trace.retrieval_embedding_text()
        assert "payments-service" in text
        assert "connection_pool_exhausted" in text
        assert "approved" in text
        assert "SYMBOLIC_FAST" in text
        # Should be compact — not the full markdown
        assert len(text) < 600

    def test_retrieval_embedding_text_includes_feedback_correction(
        self, approved_trace_with_feedback
    ):
        """When human corrected the root cause, it should appear in the retrieval text."""
        text = approved_trace_with_feedback.retrieval_embedding_text()
        assert "redis-cache" in text
        assert "Human correction" in text or "root cause was" in text

    def test_retrieval_embedding_text_no_feedback(self, approved_trace):
        """No feedback → no feedback section in embedding text."""
        text = approved_trace.retrieval_embedding_text()
        assert "Human correction" not in text

    def test_to_full_context_contains_analysis(self, approved_trace):
        """Full context should include the analysis markdown and remediation plan."""
        ctx = approved_trace.to_full_context()
        assert "payments-service" in ctx
        assert "connection_pool_exhausted" in ctx
        assert "Diagnosis" in ctx
        assert "Remediation Plan" in ctx

    def test_to_full_context_includes_feedback(self, approved_trace_with_feedback):
        """Full context should include human feedback section when present."""
        ctx = approved_trace_with_feedback.to_full_context()
        assert "Human Feedback" in ctx
        assert "modify" in ctx
        assert "partial" in ctx

    def test_to_vector_metadata_amendment5_fields(self, approved_trace):
        """Amendment #5: resolution_duration_s, symbolic_path, blast_radius_depth."""
        meta = approved_trace.to_vector_metadata()
        assert meta["resolution_duration_s"] == 142.5
        assert meta["symbolic_path"] == "SYMBOLIC_FAST"
        assert meta["blast_radius_depth"] == 2

    def test_to_vector_metadata_feedback_signals(self, approved_trace_with_feedback):
        """Feedback signals should be reflected in metadata for pre-filtering."""
        meta = approved_trace_with_feedback.to_vector_metadata()
        assert meta["has_human_feedback"] is True
        assert meta["diagnosis_accuracy"] == "partial"
        assert meta["reviewer_id"] == "sre-alice"

    def test_to_vector_metadata_no_feedback(self, approved_trace):
        """No feedback → metadata should reflect that cleanly."""
        meta = approved_trace.to_vector_metadata()
        assert meta["has_human_feedback"] is False
        assert meta["diagnosis_accuracy"] == "unknown"
        assert meta["reviewer_id"] == ""

    def test_outcome_enum_serialization(self):
        """IncidentOutcome enum values should serialize to lowercase strings."""
        assert IncidentOutcome.APPROVED.value == "approved"
        assert IncidentOutcome.REJECTED.value == "rejected"
        assert IncidentOutcome.MODIFIED.value == "modified"
        assert IncidentOutcome.AUTO_RESOLVED.value == "auto_resolved"


# ---------------------------------------------------------------------------
# HumanFeedback tests
# ---------------------------------------------------------------------------


class TestHumanFeedback:

    def test_has_new_knowledge_all_dimensions(self):
        """has_new_knowledge() returns True if any knowledge dimension is populated."""
        fb = HumanFeedback(
            decision="modify",
            new_troubleshooting_steps=["Check Redis TTL"],
        )
        assert fb.has_new_knowledge() is True

    def test_has_new_knowledge_no_knowledge(self):
        """Plain approval with no notes → has_new_knowledge() is False."""
        fb = HumanFeedback(decision="approve")
        assert fb.has_new_knowledge() is False

    def test_has_new_knowledge_diagnosis_correction(self):
        """Incorrect diagnosis with correct_root_cause → has new knowledge."""
        fb = HumanFeedback(
            decision="reject",
            diagnosis_accuracy="incorrect",
            correct_root_cause="redis-cache",
        )
        assert fb.has_new_knowledge() is True

    def test_has_new_knowledge_modification(self):
        """Modification applied → has new knowledge."""
        fb = HumanFeedback(
            decision="modify",
            modifications_applied={"replicas": 5},
        )
        assert fb.has_new_knowledge() is True

    def test_has_new_knowledge_relationship(self):
        """New system relationships → has new knowledge."""
        fb = HumanFeedback(
            decision="approve",
            new_system_relationships=["a -> b"],
        )
        assert fb.has_new_knowledge() is True

    def test_defaults_are_non_penalising(self):
        """Default values for all accuracy fields should be 'correct', not penalising."""
        fb = HumanFeedback(decision="approve")
        assert fb.diagnosis_accuracy == "correct"
        assert fb.action_appropriateness == "correct"
        assert fb.causal_path_assessment == "correct"
        assert fb.confidence_calibration == "well_calibrated"


# ---------------------------------------------------------------------------
# DocumentChunk tests
# ---------------------------------------------------------------------------


class TestDocumentChunk:

    def test_to_embedding_text_with_prefix(self, doc_chunk):
        """Embedding text should combine prefix + chunk text."""
        text = doc_chunk.to_embedding_text()
        assert doc_chunk.contextual_prefix in text
        assert doc_chunk.chunk_text in text

    def test_to_embedding_text_without_prefix(self):
        """Without prefix, embedding text equals chunk_text."""
        chunk = DocumentChunk(
            chunk_text="Check Redis OOM logs.",
            contextual_prefix="",
        )
        assert chunk.to_embedding_text() == "Check Redis OOM logs."

    def test_to_vector_metadata_amendment5_fields(self, doc_chunk):
        """Amendment #5: source_repository, last_validated, deprecated."""
        meta = doc_chunk.to_vector_metadata()
        assert meta["source_repository"] == "Corpus/Remediation"
        assert meta["deprecated"] == "False"
        assert "last_validated" in meta

    def test_deprecated_false_serialized_as_string(self, doc_chunk):
        """ChromaDB requires bool metadata as string — 'False' not False."""
        meta = doc_chunk.to_vector_metadata()
        assert isinstance(meta["deprecated"], str)
        assert meta["deprecated"] == "False"

    def test_deprecated_true_serialized_as_string(self):
        """Deprecated=True serialized as 'True' string for ChromaDB."""
        chunk = DocumentChunk(chunk_text="outdated.", deprecated=True)
        meta = chunk.to_vector_metadata()
        assert meta["deprecated"] == "True"

    def test_services_mentioned_joined(self, doc_chunk):
        """services_mentioned should be comma-joined in metadata."""
        meta = doc_chunk.to_vector_metadata()
        assert meta["services_mentioned"] == "redis-cache"


# ---------------------------------------------------------------------------
# RAGResult tests
# ---------------------------------------------------------------------------


class TestRAGResult:

    def test_get_text_for_reranking_incident(self, approved_trace):
        """Incident result returns retrieval_embedding_text for reranking."""
        result = RAGResult(
            trace=approved_trace,
            source="incident",
            similarity_score=0.82,
        )
        text = result.get_text_for_reranking()
        assert "payments-service" in text
        assert len(text) > 0

    def test_get_text_for_reranking_doc(self, doc_chunk):
        """Doc result returns to_embedding_text for reranking."""
        result = RAGResult(
            document_chunk=doc_chunk,
            source="documentation",
            similarity_score=0.75,
        )
        text = result.get_text_for_reranking()
        assert "Redis" in text

    def test_get_context_for_llm_incident(self, approved_trace):
        """LLM context for incident should include full analysis markdown."""
        result = RAGResult(trace=approved_trace, source="incident", similarity_score=0.9)
        ctx = result.get_context_for_llm()
        assert "Diagnosis" in ctx
        assert "payments-service" in ctx

    def test_get_context_for_llm_doc_uses_parent(self, doc_chunk):
        """LLM context for doc should use parent_chunk_text when available."""
        result = RAGResult(
            document_chunk=doc_chunk,
            source="documentation",
            similarity_score=0.8,
        )
        ctx = result.get_context_for_llm()
        # Should contain parent chunk content
        assert "Diagnosing Memory Pressure" in ctx or "Redis Eviction" in ctx


# ---------------------------------------------------------------------------
# RAGResponse tests
# ---------------------------------------------------------------------------


class TestRAGResponse:

    def test_to_few_shot_context_empty(self):
        """Empty results → empty context string."""
        query = RAGQuery(query_text="test query")
        response = RAGResponse(query=query, results=[])
        assert response.to_few_shot_context() == ""

    def test_to_few_shot_context_with_incidents(self, approved_trace):
        """Populated response → formatted context with section headers."""
        result = RAGResult(
            trace=approved_trace,
            source="incident",
            similarity_score=0.9,
            rerank_score=0.95,
        )
        query = RAGQuery(query_text="connection pool")
        response = RAGResponse(
            query=query,
            results=[result],
            incident_results_count=1,
            documentation_results_count=0,
        )
        ctx = response.to_few_shot_context()
        assert "RAG Context" in ctx
        assert "Historical Incident #1" in ctx
        assert "rerank=0.950" in ctx

    def test_to_few_shot_context_limits_results(self, approved_trace):
        """max_results parameter should cap the output."""
        results = [
            RAGResult(trace=approved_trace, source="incident", similarity_score=0.9 - i * 0.1)
            for i in range(5)
        ]
        query = RAGQuery(query_text="test")
        response = RAGResponse(query=query, results=results, incident_results_count=5)
        ctx = response.to_few_shot_context(max_results=2)
        # Should contain #1 and #2 but not #3
        assert "Historical Incident #2" in ctx
        assert "Historical Incident #3" not in ctx

    def test_to_few_shot_context_source_summary(self, approved_trace, doc_chunk):
        """Context footer should include source counts and fallback flag."""
        inc_result = RAGResult(trace=approved_trace, source="incident", similarity_score=0.9)
        doc_result = RAGResult(document_chunk=doc_chunk, source="documentation", similarity_score=0.7)
        query = RAGQuery(query_text="test")
        response = RAGResponse(
            query=query,
            results=[inc_result, doc_result],
            incident_results_count=1,
            documentation_results_count=1,
            hierarchical_fallback_triggered=True,
        )
        ctx = response.to_few_shot_context()
        assert "1 incident(s)" in ctx
        assert "1 documentation chunk(s)" in ctx
        assert "Fallback to docs: True" in ctx


# ---------------------------------------------------------------------------
# PII Redaction tests (Amendment #1)
# ---------------------------------------------------------------------------


class TestPIIRedaction:

    def test_redacts_ipv4(self):
        result = _redact_sensitive_data("Server at 192.168.1.100 crashed.")
        assert "192.168.1.100" not in result
        assert "[REDACTED_IP]" in result

    def test_redacts_uuid(self):
        uid = "a1b2c3d4-e5f6-4789-89ab-cdef01234567"
        result = _redact_sensitive_data(f"Request ID: {uid}")
        assert uid not in result
        assert "[REDACTED_UUID]" in result

    def test_redacts_bearer_token(self):
        result = _redact_sensitive_data("Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.abc.def")
        assert "eyJhbGciOiJSUzI1NiJ9" not in result
        assert "[REDACTED_TOKEN]" in result

    def test_redacts_api_key(self):
        result = _redact_sensitive_data("api_key=sk-prod-abc123def456ghi789")
        assert "sk-prod-abc123def456ghi789" not in result
        assert "[REDACTED_CREDENTIAL]" in result

    def test_redacts_hf_token(self):
        result = _redact_sensitive_data("Using hf_ABCDEFghijklmnopqrstuvwxyz1234 for auth")
        assert "hf_ABCDEFghijklmnopqrstuvwxyz1234" not in result
        assert "[REDACTED_HF_TOKEN]" in result

    def test_redacts_email(self):
        result = _redact_sensitive_data("Contact alice@corp-internal.io for help.")
        assert "alice@corp-internal.io" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_preserves_safe_content(self):
        safe = "Service payments-db connection pool exhausted. Restart pod."
        result = _redact_sensitive_data(safe)
        assert result == safe

    def test_empty_string_is_safe(self):
        assert _redact_sensitive_data("") == ""

    def test_none_safe_passthrough(self):
        """None-like empty input should not crash."""
        assert _redact_sensitive_data("   ") == "   "


# ---------------------------------------------------------------------------
# ConfidenceEngine tests
# ---------------------------------------------------------------------------


class TestConfidenceEngine:

    def test_composite_high_confidence_no_hitl(self):
        """High diagnosis + high RAG + all policy approved → above threshold."""
        engine = ConfidenceEngine(threshold=0.75)
        score = engine.compute_composite_confidence(
            diagnosis_confidence=0.90,
            rag_boost=0.20,
            policy_pass_rate=1.0,
        )
        assert score > 0.75
        assert engine.requires_hitl(score) is False

    def test_composite_low_confidence_requires_hitl(self):
        """Low diagnosis + no RAG + none approved → below threshold."""
        engine = ConfidenceEngine(threshold=0.75)
        score = engine.compute_composite_confidence(
            diagnosis_confidence=0.50,
            rag_boost=0.0,
            policy_pass_rate=0.0,
        )
        assert score < 0.75
        assert engine.requires_hitl(score) is True

    def test_composite_score_clipped_to_1(self):
        """Composite score should never exceed 1.0."""
        engine = ConfidenceEngine(threshold=0.75)
        score = engine.compute_composite_confidence(
            diagnosis_confidence=1.0,
            rag_boost=0.25,
            policy_pass_rate=1.0,
        )
        assert score <= 1.0

    def test_composite_score_weights_sum(self):
        """Check that W_DIAG + W_RAG + W_POLICY = 1.0."""
        engine = ConfidenceEngine()
        total = engine.W_DIAG + engine.W_RAG + engine.W_POLICY
        assert abs(total - 1.0) < 1e-9

    def test_rag_boost_normalisation(self):
        """Max RAG boost (0.25) normalised to 1.0 should contribute full W_RAG weight."""
        engine = ConfidenceEngine(threshold=0.75)
        # Perfect diagnosis, max RAG, no policy approval
        score_max_rag = engine.compute_composite_confidence(
            diagnosis_confidence=0.0,
            rag_boost=0.25,
            policy_pass_rate=0.0,
        )
        # Perfect diagnosis, no RAG, no policy approval
        score_no_rag = engine.compute_composite_confidence(
            diagnosis_confidence=0.0,
            rag_boost=0.0,
            policy_pass_rate=0.0,
        )
        # The difference should equal W_RAG
        assert abs((score_max_rag - score_no_rag) - engine.W_RAG) < 1e-6

    def test_policy_pass_rate_from_decisions(self):
        """compute_policy_pass_rate should correctly count AUTO_APPROVED decisions."""
        from airs_v2.action.types import RiskLevel
        engine = ConfidenceEngine()
        decisions = [
            PolicyDecision(
                action_id=f"a{i}",
                action_kind="restart_pod",
                effective_risk_level=RiskLevel.LOW,
                approved=i < 3,
                decision=DecisionOutcome.AUTO_APPROVED if i < 3 else DecisionOutcome.PENDING_HUMAN,
                violations=[],
            )
            for i in range(4)
        ]
        rate = engine.compute_policy_pass_rate(decisions)
        assert rate == 0.75  # 3 of 4 approved

    def test_policy_pass_rate_empty_decisions(self):
        """Empty decisions list → 0.0 pass rate."""
        engine = ConfidenceEngine()
        assert engine.compute_policy_pass_rate([]) == 0.0


# ---------------------------------------------------------------------------
# FeedbackCollector tests
# ---------------------------------------------------------------------------


class TestFeedbackCollector:

    def test_parse_cli_input_minimal(self):
        """Minimal JSON with just decision should parse cleanly."""
        collector = FeedbackCollector()
        json_str = json.dumps({"decision": "approve"})
        fb = collector.parse_cli_input(json_str)
        assert fb.decision == "approve"
        assert fb.diagnosis_accuracy == "correct"

    def test_parse_cli_input_full(self):
        """Full JSON should populate all provided fields."""
        collector = FeedbackCollector()
        payload = {
            "decision": "modify",
            "diagnosis_accuracy": "partial",
            "correct_root_cause": "redis-cache",
            "domain_expertise_notes": "Redis was the trigger.",
            "new_system_relationships": ["a -> b"],
        }
        fb = collector.parse_cli_input(json.dumps(payload))
        assert fb.decision == "modify"
        assert fb.diagnosis_accuracy == "partial"
        assert fb.correct_root_cause == "redis-cache"
        assert fb.domain_expertise_notes == "Redis was the trigger."
        assert fb.new_system_relationships == ["a -> b"]

    def test_parse_cli_input_unknown_keys_ignored(self):
        """Unknown JSON keys should not crash the parser."""
        collector = FeedbackCollector()
        json_str = json.dumps({"decision": "approve", "unknown_field": "value"})
        fb = collector.parse_cli_input(json_str)
        assert fb.decision == "approve"

    def test_create_auto_feedback_approve(self):
        """Auto-feedback with approve=True should produce approve decision."""
        collector = FeedbackCollector()
        fb = collector.create_auto_feedback(auto_approve=True, reviewer_id="bot")
        assert fb.decision == "approve"
        assert fb.reviewer_id == "bot"
        assert fb.diagnosis_accuracy == "correct"

    def test_create_auto_feedback_reject(self):
        """Auto-feedback with approve=False should produce reject decision."""
        collector = FeedbackCollector()
        fb = collector.create_auto_feedback(auto_approve=False)
        assert fb.decision == "reject"

    def test_parse_invalid_json_raises(self):
        """Invalid JSON should raise json.JSONDecodeError."""
        import json as _json
        collector = FeedbackCollector()
        with pytest.raises(_json.JSONDecodeError):
            collector.parse_cli_input("not-json")


# ---------------------------------------------------------------------------
# NetworkXGraphStore tests
# ---------------------------------------------------------------------------


class TestNetworkXGraphStore:

    def test_ingest_trace_creates_nodes(self, graph_store, approved_trace):
        """Ingesting a trace should create incident, service, and pattern nodes."""
        graph_store.ingest_trace(approved_trace)
        stats = graph_store.stats()
        assert stats["nodes"] > 0
        assert "incident" in stats["node_types"]
        assert "service" in stats["node_types"]
        assert "pattern" in stats["node_types"]

    def test_ingest_trace_service_history(self, graph_store, approved_trace):
        """After ingest, service incident history should include the trace."""
        graph_store.ingest_trace(approved_trace)
        history = graph_store.get_service_incident_history("payments-service", limit=10)
        assert len(history) >= 1
        assert any(h["trace_id"] == "trace-001" for h in history)

    def test_ingest_multiple_traces_pattern_count(self, graph_store, approved_trace):
        """Ingesting 3 traces with same incident_type should give pattern count=3."""
        for i in range(3):
            t = approved_trace.model_copy(
                update={"trace_id": f"trace-00{i+1}"}
            )
            graph_store.ingest_trace(t)
        pattern = graph_store.get_pattern_frequency("connection_pool_exhausted")
        assert pattern["count"] == 3

    def test_get_pattern_frequency_unknown(self, graph_store):
        """Unknown template key → count=0, empty outcomes."""
        result = graph_store.get_pattern_frequency("unknown_pattern")
        assert result["count"] == 0
        assert result["outcomes"] == {}

    def test_ingest_feedback_misdiagnosis_edge(self, graph_store, approved_trace):
        """Incorrect diagnosis feedback should create MISDIAGNOSED_AS edge."""
        graph_store.ingest_trace(approved_trace)
        feedback = HumanFeedback(
            decision="reject",
            diagnosis_accuracy="incorrect",
            correct_root_cause="redis-cache",
        )
        graph_store.ingest_feedback(approved_trace.trace_id, feedback)

        misdiagnoses = graph_store.get_misdiagnosis_history("payments-service")
        assert len(misdiagnoses) >= 1
        assert misdiagnoses[0]["actual_root_cause"] == "redis-cache"

    def test_graph_persistence(self, tmp_path, approved_trace):
        """Graph should persist to and reload from JSON correctly."""
        persist_path = tmp_path / "persist_test.json"
        store1 = NetworkXGraphStore(persist_path=str(persist_path))
        store1.ingest_trace(approved_trace)
        store1.close()

        # Reload
        store2 = NetworkXGraphStore(persist_path=str(persist_path))
        stats = store2.stats()
        assert stats["nodes"] > 0

    def test_get_service_incident_history_empty(self, graph_store):
        """Unknown service → empty history list."""
        history = graph_store.get_service_incident_history("nonexistent-svc")
        assert history == []

    def test_get_causal_chains_empty(self, graph_store):
        """Unknown service → empty causal chains."""
        chains = graph_store.get_causal_chains("nonexistent-svc")
        assert chains == []

    def test_ingest_feedback_new_relationships(self, graph_store, approved_trace):
        """New system relationships from feedback should create service edges in graph."""
        graph_store.ingest_trace(approved_trace)
        feedback = HumanFeedback(
            decision="approve",
            new_system_relationships=["payments-service -> redis-cache"],
        )
        graph_store.ingest_feedback(approved_trace.trace_id, feedback)
        stats = graph_store.stats()
        # redis-cache should now be a service node
        assert stats["nodes"] > 3  # original nodes + new redis-cache node
