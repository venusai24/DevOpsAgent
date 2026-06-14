"""
airs_v2/memory/rag_engine.py
==============================

Top-level RAG orchestrator — the public API of the memory subsystem.

Responsibilities
----------------
1. Hierarchical retrieval (incidents first → docs fallback)
2. Cross-encoder reranking (N candidates → K final results)
3. Knowledge graph context enrichment (causal chains, misdiagnosis history)
4. RAG confidence boost computation for the ConfidenceEngine
5. Context assembly for LLM injection (respects MAX_RAG_CONTEXT_TOKENS)

Phase isolation invariant
--------------------------
retrieve_diagnosis_context() is called ONLY by ReasoningEngine (Stage 3).
retrieve_action_context()    is called ONLY by ExecutionEngine (Stage 4).
The RAG engine itself never triggers HITL — it is purely an inference tool.

Amendment #4 — Context token guard:
  to_few_shot_context() limits output to MAX_RAG_CONTEXT_TOKENS.
  Phase 2 will apply LLMLingua-2 when context exceeds this threshold.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Hybrid Hierarchical RAG Engine.

    Parameters
    ----------
    vector_store  : VectorStore instance (auto-created if None).
    graph_store   : GraphStore instance (auto-created if None).
    reranker      : CrossEncoderReranker instance (auto-created if None).
    """

    def __init__(
        self,
        vector_store=None,
        graph_store=None,
        reranker=None,
    ) -> None:
        from config import settings
        from airs_v2.memory.vector_store import VectorStore
        from airs_v2.memory.graph_store import NetworkXGraphStore
        from airs_v2.memory.reranker import CrossEncoderReranker

        self._vector_store = vector_store or VectorStore()
        self._graph_store = graph_store or NetworkXGraphStore()
        self._reranker = reranker or (
            CrossEncoderReranker() if settings.RAG_RERANKING_ENABLED else None
        )
        self._settings = settings
        logger.info(
            "[RAGEngine] Initialised. incidents=%d docs=%d reranking=%s",
            self._vector_store.incident_count(),
            self._vector_store.doc_count(),
            self._reranker is not None,
        )

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def graph_store(self):
        return self._graph_store

    # ── Public API: Diagnosis Phase ────────────────────────────────────────────

    def retrieve_diagnosis_context(
        self,
        analysis_markdown: str,
        focal_service: str,
        incident_type: str,
        top_k: Optional[int] = None,
    ):  # → RAGResponse
        """
        Hierarchical retrieval for the autonomous Diagnosis Phase (Stage 3).

        The query is constructed from the live analysis markdown, focal service,
        and incident type — not raw logs. This ensures the vector search operates
        on semantic incident descriptions, not literal log payloads.
        """
        from airs_v2.memory.types import RAGQuery, RAGResponse

        top_k = top_k or self._settings.RAG_TOP_K
        query_text = self._build_diagnosis_query(
            analysis_markdown, focal_service, incident_type
        )

        return self._run_retrieval(
            query_text=query_text,
            focal_service=focal_service,
            incident_type=incident_type,
            top_k=top_k,
            include_documentation=True,
            phase="diagnosis",
        )

    # ── Public API: Action Phase ───────────────────────────────────────────────

    def retrieve_action_context(
        self,
        analysis_markdown: str,
        focal_service: str,
        incident_type: str,
        top_k: int = 3,
    ):  # → RAGResponse
        """
        Focused retrieval for the Action Phase (Stage 4).

        Queries approved incident traces to surface past successful remediations.
        Documentation is also searched for relevant runbook procedures.
        """
        query_text = self._build_action_query(
            analysis_markdown, focal_service, incident_type
        )

        return self._run_retrieval(
            query_text=query_text,
            focal_service=focal_service,
            incident_type=incident_type,
            top_k=top_k,
            include_documentation=True,
            phase="action",
        )

    # ── Public API: Confidence Boost ───────────────────────────────────────────

    def compute_rag_confidence_boost(self, response) -> float:  # response: RAGResponse
        """
        Compute a RAG-derived confidence boost in [0.0, 0.25].

        Three factors contribute:
          1. Context precision (rerank scores > 0)    weight 0.10
          2. Top result similarity                    weight 0.10
          3. Episodic density (approved result count) weight 0.05

        Returns a boost value that is added to the base diagnosis confidence
        by the ConfidenceEngine.
        """
        if not response.results:
            return 0.0

        # Factor 1: context precision (fraction with non-zero rerank scores)
        precision = response.context_precision  # Already computed in _run_retrieval()
        precision_boost = precision * 0.10

        # Factor 2: top result similarity
        top_similarity = response.results[0].similarity_score if response.results else 0.0
        similarity_boost = top_similarity * 0.10

        # Factor 3: episodic density — approved incident results
        approved_count = min(response.incident_results_count, 5)
        density_boost = (approved_count / 5.0) * 0.05

        total_boost = round(precision_boost + similarity_boost + density_boost, 4)
        logger.debug(
            "[RAGEngine] Confidence boost: precision=%.3f similarity=%.3f density=%.3f → total=%.4f",
            precision_boost,
            similarity_boost,
            density_boost,
            total_boost,
        )
        return total_boost

    # ── Core retrieval pipeline ────────────────────────────────────────────────

    def _run_retrieval(
        self,
        query_text: str,
        focal_service: str,
        incident_type: str,
        top_k: int,
        include_documentation: bool,
        phase: str,
    ):  # → RAGResponse
        from airs_v2.memory.types import RAGQuery, RAGResponse

        query = RAGQuery(
            query_text=query_text,
            focal_service=focal_service,
            incident_type=incident_type,
            top_k=top_k,
            candidate_k=self._settings.RAG_CANDIDATE_K,
            min_similarity=self._settings.RAG_MIN_SIMILARITY,
            exclude_rejected=True,
            exclude_deprecated_docs=True,
            include_documentation=include_documentation,
        )

        t0 = time.perf_counter()

        # Stage 1: Hierarchical vector retrieval
        candidates, fallback_triggered = self._vector_store.query_hierarchical(query)
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000

        incident_count = sum(1 for r in candidates if r.source == "incident")
        doc_count = sum(1 for r in candidates if r.source == "documentation")

        logger.debug(
            "[RAGEngine] %s phase: %d candidates (%d incidents, %d docs) in %.1fms",
            phase,
            len(candidates),
            incident_count,
            doc_count,
            retrieval_latency_ms,
        )

        # Stage 2: Cross-encoder reranking
        reranking_latency_ms = 0.0
        if self._reranker and candidates:
            final_results, reranking_latency_ms = self._reranker.rerank(
                query=query_text,
                results=candidates,
                top_k=top_k,
            )
        else:
            # No reranking: sort by similarity and truncate
            final_results = sorted(
                candidates, key=lambda r: r.similarity_score, reverse=True
            )[:top_k]

        # Enrich with graph context
        graph_context = self._build_graph_context(focal_service)

        # Compute context precision
        with_rerank = [r for r in final_results if r.rerank_score is not None]
        context_precision = len(with_rerank) / len(final_results) if final_results else 0.0

        # Recount by source after reranking
        final_incident_count = sum(1 for r in final_results if r.source == "incident")
        final_doc_count = sum(1 for r in final_results if r.source == "documentation")

        response = RAGResponse(
            query=query,
            results=final_results,
            incident_results_count=final_incident_count,
            documentation_results_count=final_doc_count,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            reranking_latency_ms=round(reranking_latency_ms, 2),
            context_precision=round(context_precision, 4),
            hierarchical_fallback_triggered=fallback_triggered,
        )

        total_ms = retrieval_latency_ms + reranking_latency_ms
        logger.info(
            "[RAGEngine] %s retrieval done: %d results in %.1fms total (fallback=%s)",
            phase,
            len(final_results),
            total_ms,
            fallback_triggered,
        )
        return response

    # ── Graph context enrichment ───────────────────────────────────────────────

    def _build_graph_context(self, focal_service: str) -> str:
        """
        Build a compact graph context string to append to the RAG prompt.

        Includes:
          - Recent incident history for the focal service
          - Known causal chains from the focal service
          - Misdiagnosis warnings (if this service has been wrongly blamed before)
        """
        lines: list[str] = []

        try:
            history = self._graph_store.get_service_incident_history(
                focal_service, limit=3
            )
            if history:
                lines.append(f"**Graph: {focal_service} incident history (last 3)**")
                for h in history:
                    lines.append(
                        f"- {h['incident_type']} → {h['outcome']} "
                        f"(conf={h['diagnosis_confidence']:.2f})"
                    )

            chains = self._graph_store.get_causal_chains(focal_service, depth=2)
            if chains:
                lines.append(f"**Graph: Known causal chains from {focal_service}**")
                for chain in chains[:3]:
                    lines.append(f"- {' → '.join(chain)}")

            misdiagnoses = self._graph_store.get_misdiagnosis_history(focal_service)
            if misdiagnoses:
                lines.append(
                    f"⚠️ **Graph: {focal_service} has been wrongly blamed {misdiagnoses[0]['count']}x "
                    f"— actual root cause was '{misdiagnoses[0]['actual_root_cause']}'.**"
                )
        except Exception as exc:
            logger.warning("[RAGEngine] Graph context enrichment failed: %s", exc)

        return "\n".join(lines)

    # ── Query builders ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_diagnosis_query(
        analysis_markdown: str, focal_service: str, incident_type: str
    ) -> str:
        """Construct a focused semantic query for diagnosis-phase retrieval."""
        # Use the first 800 chars of analysis to capture the most diagnostic content
        summary = (analysis_markdown or "")[:800].strip()
        parts = [
            f"Service: {focal_service}.",
            f"Incident type: {incident_type}.",
        ]
        if summary:
            parts.append(f"Context: {summary}")
        return " ".join(parts)

    @staticmethod
    def _build_action_query(
        analysis_markdown: str, focal_service: str, incident_type: str
    ) -> str:
        """Construct a focused semantic query for action-phase retrieval."""
        summary = (analysis_markdown or "")[:400].strip()
        parts = [
            f"Remediation for {focal_service}.",
            f"Failure pattern: {incident_type}.",
            "What is the correct declarative remediation action?",
        ]
        if summary:
            parts.append(f"Incident context: {summary}")
        return " ".join(parts)
