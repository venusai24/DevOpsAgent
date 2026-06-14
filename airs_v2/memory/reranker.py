"""
airs_v2/memory/reranker.py
===========================

Cross-encoder reranking service — Stage 2 of the two-stage retrieval pipeline.

Purpose
-------
Bi-encoders (Stage 1 vector search) generate query and document embeddings
independently. They capture broad semantic similarity but miss subtle interaction
signals (negations, specific intent, technical precision). Cross-encoders process
the query and document *together* with full cross-attention, producing precise
relevance scores at the cost of higher latency.

This implements the N >> K principle:
  Stage 1: Broad retrieval → N = 50 candidates (cheap, parallel)
  Stage 2: Cross-encoder rerank → K = 5 final results (precise)

Model selection (Amendment #3)
--------------------------------
The reranker model is configurable via config.py RERANKER_MODEL:
  Phase 1 default  : cross-encoder/ms-marco-MiniLM-L-6-v2  (CPU, ~50ms, English)
  Phase 2 upgrade  : BAAI/bge-reranker-v2-m3               (GPU, ~100ms, multilingual)

Both are point-wise cross-encoders. The BGE model provides measurably higher
nDCG@10 on technical retrieval benchmarks when GPU is available.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Two-stage retrieval: Stage 2 precision engine.

    Lazy-loads the cross-encoder model on first call to avoid import-time
    overhead when reranking is disabled.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        from config import settings

        self._model_name = model_name or settings.RERANKER_MODEL or _DEFAULT_MODEL
        self._model = None          # Lazy-loaded
        logger.info(
            "[CrossEncoderReranker] Initialised model=%s", self._model_name
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info(
            "[CrossEncoderReranker] Loading model %s …", self._model_name
        )
        self._model = CrossEncoder(self._model_name)
        logger.info("[CrossEncoderReranker] Model loaded.")

    def rerank(
        self,
        query: str,
        results: list,          # list[RAGResult]
        top_k: int = 5,
    ) -> tuple[list, float]:    # (reranked_results, latency_ms)
        """
        Rerank a list of RAGResult objects using full cross-attention.

        Parameters
        ----------
        query     : The original retrieval query string.
        results   : Candidate RAGResult objects from Stage 1 retrieval.
        top_k     : Number of top results to return after reranking.

        Returns
        -------
        tuple of (reranked_results[:top_k], latency_ms)
        """
        if not results:
            return [], 0.0

        self._load_model()

        t0 = time.perf_counter()
        candidate_texts = [r.get_text_for_reranking() for r in results]
        pairs = [[query, text] for text in candidate_texts]

        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Attach rerank scores and sort descending
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        reranked = sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)

        logger.debug(
            "[CrossEncoderReranker] Reranked %d candidates → top %d in %.1fms",
            len(results),
            top_k,
            latency_ms,
        )
        return reranked[:top_k], latency_ms
