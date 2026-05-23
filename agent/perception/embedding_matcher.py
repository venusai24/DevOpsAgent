"""
agent/perception/embedding_matcher.py

L2 Retrieval-Assisted Matching — TF-IDF vector similarity for log classification.

When a log entry fails the L1 regex match, L2 computes a TF-IDF embedding
of the log text and compares it against pre-computed embeddings of all known
log patterns. If cosine similarity exceeds 0.75, the entry is classified
without invoking the L3 LLM.

Uses scikit-learn's TfidfVectorizer — no external API calls required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agent.perception.template_cache import _BUILTIN_TEMPLATES

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    np = None  # type: ignore
    HAS_SKLEARN = False
    logger.warning("[L2Matcher] scikit-learn not installed — L2 matching disabled, all unknowns go to L3.")

_L2_THRESHOLD = 0.75  # Minimum cosine similarity for L2 auto-classification


@dataclass
class L2Match:
    """Result of an L2 TF-IDF similarity lookup."""
    template_key: str
    similarity: float
    tier: str = "L2"


class EmbeddingMatcher:
    """
    L2 TF-IDF embedding matcher for log classification.

    Pre-computes TF-IDF vectors for all known log pattern descriptions at
    initialization, then classifies new log entries by cosine similarity.

    Falls back gracefully when scikit-learn is unavailable.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._vectorizer: Optional[Any] = None
        self._pattern_matrix: Optional[Any] = None
        self._pattern_keys: list[str] = []
        self._l2_hits: int = 0
        self._l2_misses: int = 0

    def fit(self, templates: list[tuple[str, str]] = None) -> None:
        """
        Fit the TF-IDF vectorizer against the known template corpus.

        Args:
            templates: List of (key, pattern_regex) tuples. Defaults to _BUILTIN_TEMPLATES.
        """
        if not HAS_SKLEARN:
            return
        templates = templates or _BUILTIN_TEMPLATES
        self._pattern_keys = [key for key, _ in templates]
        # Use the regex pattern itself as the document text (removes regex special chars first)
        import re
        corpus = [
            re.sub(r"[\\^$.*+?()\[\]{}|]", " ", pattern)
            for _, pattern in templates
        ]
        self._vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        self._pattern_matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True
        logger.info("[L2Matcher] Fitted TF-IDF on %d templates.", len(templates))

    def find_nearest(self, log_entry: str) -> Optional[L2Match]:
        """
        Find the most similar known template for an unrecognized log entry.

        Args:
            log_entry: The raw log line to classify.

        Returns:
            L2Match if cosine similarity >= _L2_THRESHOLD, else None (L3 needed).
        """
        if not self._fitted:
            self.fit()
        if not HAS_SKLEARN or self._vectorizer is None:
            return None
        try:
            query_vec = self._vectorizer.transform([log_entry])
            similarities = cosine_similarity(query_vec, self._pattern_matrix)[0]
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])
            if best_sim >= _L2_THRESHOLD:
                self._l2_hits += 1
                return L2Match(
                    template_key=self._pattern_keys[best_idx],
                    similarity=round(best_sim, 3),
                )
            self._l2_misses += 1
            return None
        except Exception as exc:
            logger.warning("[L2Matcher] Similarity computation failed: %s", exc)
            return None

    @property
    def stats(self) -> dict:
        """Return L2 hit/miss statistics."""
        return {"l2_hits": self._l2_hits, "l2_misses": self._l2_misses, "fitted": self._fitted}
