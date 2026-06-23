"""
ConANN Retrieval Engine — Module 1.8.

Conformal Adaptive Nearest Neighbor (ConANN) retrieval over the 6 Qdrant
knowledge collections. Provides marginal coverage guarantees via conformal
prediction: for a query q, retrieved documents satisfy:

    P(nonconformity(q, d) ≤ q̂_{1-α}) ≥ 1 - α

Algorithm per hop:
  1. Embed the query text
  2. Run initial ANN search with k = typical_k
  3. Compute nonconformity scores: s_i = 1 - cosim(q, d)
  4. Adaptive expansion: if any s_i > q̂, expand search radius (increase k)
  5. Repeat until all retrieved docs satisfy s_i ≤ q̂, or max_expansions reached
  6. Return only the conforming subset

Phase 1: q̂ values are hardcoded in calibration/defaults.json.
Phase 3: q̂ values computed by scripts/calibrate.py (synthetic bootstrap).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, ScoredPoint

from airs.retrieval.collections import build_payload_filter
from airs.retrieval.embedding import EmbeddingService, get_embedding_service
from airs.risk.calibration import CalibrationStore, get_calibration_store

log = logging.getLogger(__name__)


class ConANNResult:
    """Result from a ConANN retrieval query."""
    __slots__ = ("document_id", "collection", "score", "nonconformity", "payload", "content")

    def __init__(
        self,
        document_id: str,
        collection: str,
        score: float,
        nonconformity: float,
        payload: dict[str, Any],
        content: str,
    ) -> None:
        self.document_id = document_id
        self.collection = collection
        self.score = score                  # Cosine similarity ∈ [0, 1]
        self.nonconformity = nonconformity  # s_i = 1 - score ∈ [0, 1]
        self.payload = payload
        self.content = content              # Full document text (from payload)

    def __repr__(self) -> str:
        return (
            f"ConANNResult(doc={self.document_id!r}, "
            f"score={self.score:.3f}, nonconformity={self.nonconformity:.3f})"
        )


class ConANNRetriever:
    """
    Conformal Adaptive Nearest Neighbor retrieval engine.

    Wraps Qdrant with adaptive k-expansion and conformal coverage enforcement.
    One retriever instance is shared across all Temporal activities (stateless).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_service: EmbeddingService,
        calibration_store: CalibrationStore,
    ) -> None:
        self._qdrant = qdrant_client
        self._embedder = embedding_service
        self._calibration = calibration_store

    @classmethod
    def from_settings(cls) -> "ConANNRetriever":
        """Create ConANNRetriever from global singletons."""
        from airs.config import settings
        return cls(
            qdrant_client=QdrantClient(url=settings.qdrant_url),
            embedding_service=get_embedding_service(),
            calibration_store=get_calibration_store(),
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        collection: str,
        query_text: str,
        k: Optional[int] = None,
        payload_filter: Optional[Filter] = None,
        trigger_state: Optional[str] = None,
        failure_domain: Optional[str] = None,
    ) -> list[ConANNResult]:
        """
        Run a conformal adaptive retrieval query.

        Args:
            collection:     One of the 6 Qdrant collection names.
            query_text:     Natural language query to embed.
            k:              Override for number of results (default: typical_k from calibration).
            payload_filter: Pre-built Qdrant filter (takes precedence over keyword args).
            trigger_state:  Optional trigger state pre-filter (e.g., 'Continue').
            failure_domain: Optional failure domain pre-filter (e.g., 'data_tier').

        Returns:
            List of ConANNResult, all satisfying s_i ≤ q̂_{1-α}.
            Empty list if collection is empty or no conforming results found.
        """
        # Load calibration for this collection
        q_hat = self._calibration.get_q_hat(collection)
        alpha = self._calibration.get_collection_alpha(collection)
        max_expansions = self._calibration.get_max_expansions(collection)
        typical_k = k or self._calibration.get_typical_k(collection)

        log.debug(
            "ConANN query: collection=%s q̂=%.3f α=%.2f initial_k=%d",
            collection, q_hat, alpha, typical_k,
        )

        # Build payload filter
        if payload_filter is None and (trigger_state or failure_domain):
            payload_filter = build_payload_filter(
                trigger_state=trigger_state,
                failure_domain=failure_domain,
            )

        # Embed query
        try:
            query_vec = self._embedder.embed_single(query_text)
        except Exception as e:
            log.error("Embedding failed for ConANN query: %s", e)
            return []

        # Adaptive k-expansion loop
        current_k = typical_k
        conforming_results: list[ConANNResult] = []

        for expansion in range(max_expansions + 1):
            try:
                raw_results = self._qdrant.search(
                    collection_name=collection,
                    query_vector=query_vec,
                    limit=current_k,
                    query_filter=payload_filter,
                    with_payload=True,
                )
            except Exception as e:
                log.error("Qdrant search failed for collection %s: %s", collection, e)
                return []

            if not raw_results:
                log.debug("ConANN: empty result set from Qdrant (k=%d)", current_k)
                break

            # Compute nonconformity scores
            conforming: list[ConANNResult] = []
            non_conforming_count = 0

            for point in raw_results:
                cosim = float(point.score)  # Qdrant cosine score ∈ [−1, 1] → normalized [0,1]
                # Qdrant COSINE distance uses 1 - cosim internally, score returned is similarity
                nonconformity = max(0.0, 1.0 - cosim)

                result = ConANNResult(
                    document_id=str(point.payload.get("document_id", point.id)),
                    collection=collection,
                    score=cosim,
                    nonconformity=nonconformity,
                    payload=point.payload or {},
                    content=str(point.payload.get("content_preview", ""))
                    if point.payload else "",
                )

                if nonconformity <= q_hat:
                    conforming.append(result)
                else:
                    non_conforming_count += 1

            log.debug(
                "ConANN expansion %d: k=%d conforming=%d non-conforming=%d",
                expansion, current_k, len(conforming), non_conforming_count,
            )

            if non_conforming_count == 0 or expansion == max_expansions:
                # All retrieved results conform — done
                conforming_results = conforming
                break

            # Expand k for next iteration
            current_k = min(current_k * 2, 50)

        log.debug(
            "ConANN final: collection=%s, %d conforming results returned",
            collection, len(conforming_results),
        )
        return conforming_results

    def retrieve_for_intent(
        self,
        collection: str,
        query_text: str,
        filters: dict[str, Any],
        k: int = 3,
    ) -> list[ConANNResult]:
        """
        Simplified interface for activity-level retrieval with dict-based filters.

        Args:
            collection:  Target collection.
            query_text:  Embedded query.
            filters:     Dict of {field: value} pre-filters.
            k:           Initial number of results.

        Returns:
            Conforming ConANNResults.
        """
        payload_filter = build_payload_filter(
            trigger_state=filters.get("trigger_state"),
            failure_domain=filters.get("failure_domain"),
            tool_tier=filters.get("tool_tier"),
            knowledge_type=filters.get("knowledge_type"),
        )
        return self.retrieve(
            collection=collection,
            query_text=query_text,
            k=k,
            payload_filter=payload_filter,
        )

    def retrieve_diagnostic_context(
        self,
        query_text: str,
        failure_domain: Optional[str] = None,
    ) -> list[ConANNResult]:
        """Convenience: retrieve from C1 diagnostic_knowledge."""
        return self.retrieve(
            collection="diagnostic_knowledge",
            query_text=query_text,
            failure_domain=failure_domain,
        )

    def retrieve_remediation(
        self,
        query_text: str,
        trigger_state: str = "Diagnose",
    ) -> list[ConANNResult]:
        """Convenience: retrieve from C4 remediation_actions."""
        return self.retrieve(
            collection="remediation_actions",
            query_text=query_text,
            trigger_state=trigger_state,
        )

    def retrieve_failure_signature(
        self,
        query_text: str,
        failure_domain: Optional[str] = None,
    ) -> list[ConANNResult]:
        """Convenience: retrieve from C5 failure_signatures."""
        return self.retrieve(
            collection="failure_signatures",
            query_text=query_text,
            failure_domain=failure_domain,
        )

    def retrieve_operational_constraints(
        self,
        query_text: str,
    ) -> list[ConANNResult]:
        """Convenience: retrieve from C6 operational_constraints."""
        return self.retrieve(
            collection="operational_constraints",
            query_text=query_text,
        )


# ─── Module-level singleton ────────────────────────────────────────────────────

_retriever: Optional[ConANNRetriever] = None


def get_conann_retriever() -> ConANNRetriever:
    """Return the global ConANNRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = ConANNRetriever.from_settings()
    return _retriever
