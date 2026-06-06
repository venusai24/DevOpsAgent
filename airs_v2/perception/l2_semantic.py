"""
airs_v2/perception/l2_semantic.py
==================================

L2 — Semantic Retrieval-Assisted Matching
------------------------------------------
The second tier in the NeSy-Edge log parsing router.

When a log entry fails the L1 regex cache (i.e., it does not match any
known pattern exactly), L2 computes a **TF-IDF vector** of the log text and
compares it against a pre-built **knowledge base** of indexed log pattern
descriptions via cosine similarity.

A match is accepted when ``similarity >= threshold`` (default 0.72).

Improvements over v1 (agent/perception/embedding_matcher.py)
-------------------------------------------------------------
* **Knowledge base is built from descriptions, not raw regexes** — the
  human-readable description ("OOM killer evicted process") is a far richer
  TF-IDF document than the raw regex string.
* **Configurable threshold** — passed at construction; allows tests to tune
  the boundary without monkey-patching a module-level constant.
* **Incremental ``add_entry``** — when L3 learns a new pattern, L2 can
  absorb it without a full re-fit (vocabulary is extended via a new
  TfidfVectorizer fitted on the augmented corpus).
* **Top-k candidates returned** — ``find_nearest_k`` exposes ranked
  candidates for diagnostic visibility (used in ``RouterResult``).
* **Graceful degradation** — if scikit-learn is not installed the instance
  silently passes every query to L3; no import error at module load.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn import
# ---------------------------------------------------------------------------

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SKLEARN = True
except ImportError:
    np = None  # type: ignore
    TfidfVectorizer = None  # type: ignore
    sk_cosine = None  # type: ignore
    _HAS_SKLEARN = False
    logger.warning(
        "[L2Semantic] scikit-learn not installed — L2 tier disabled; "
        "all non-L1 logs will go directly to L3."
    )

_REGEX_NOISE = re.compile(r"[\\^$.*+?()\[\]{}|]")


@dataclass
class L2Match:
    """Result of a successful L2 semantic similarity lookup."""
    template_key: str
    description: str
    similarity: float
    tier: str = "L2"


@dataclass
class _KBEntry:
    """A single record in the L2 knowledge base."""
    key: str
    description: str          # Human-readable description (main TF-IDF document)
    pattern_hint: str = ""    # Cleaned regex (appended to description for extra signal)


class L2SemanticMatcher:
    """
    TF-IDF cosine-similarity matcher over a knowledge base of log pattern
    descriptions.

    Lifecycle
    ---------
    1. Construct with an optional ``threshold`` and ``knowledge_base``.
    2. ``fit()`` (or pass ``auto_fit=True``) pre-computes the TF-IDF matrix.
    3. ``find_nearest(log_entry)`` returns an ``L2Match`` or ``None``.
    4. ``add_entry(...)`` extends the knowledge base and triggers a re-fit.

    The knowledge base defaults to the descriptions in
    ``l1_cache.BUILTIN_TEMPLATES``; it is automatically extended when L3
    promotes a novel template to L1.
    """

    DEFAULT_THRESHOLD = 0.72

    def __init__(
        self,
        knowledge_base: Optional[list[_KBEntry]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        auto_fit: bool = True,
    ) -> None:
        self._threshold = threshold
        self._kb: list[_KBEntry] = knowledge_base or self._default_kb()
        self._fitted = False
        self._vectorizer: Optional[object] = None
        self._matrix: Optional[object] = None  # shape (n_docs, n_features)

        self._l2_hits = 0
        self._l2_misses = 0

        if auto_fit and _HAS_SKLEARN:
            self.fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """(Re)fit the TF-IDF vectorizer on the current knowledge base corpus."""
        if not _HAS_SKLEARN:
            return
        corpus = [self._doc(entry) for entry in self._kb]
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,   # log(1+tf) — reduces weight of very common terms
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True
        logger.debug("[L2Semantic] Fitted TF-IDF on %d entries.", len(self._kb))

    def find_nearest(self, log_entry: str) -> Optional[L2Match]:
        """
        Find the most similar knowledge-base entry for an unrecognised log.

        Returns an ``L2Match`` when ``similarity >= threshold``, else ``None``
        (L3 LLM fallback required).
        """
        candidates = self.find_nearest_k(log_entry, k=1)
        if candidates:
            return candidates[0]
        return None

    def find_nearest_k(self, log_entry: str, k: int = 3) -> list[L2Match]:
        """
        Return the top-k nearest knowledge-base entries above the threshold.
        Used for diagnostic visibility in RouterResult.
        """
        if not _HAS_SKLEARN or not self._fitted or self._vectorizer is None:
            self._l2_misses += 1
            return []
        if not log_entry.strip():
            self._l2_misses += 1
            return []
        try:
            query_vec = self._vectorizer.transform([log_entry])  # type: ignore[union-attr]
            sims: list[float] = sk_cosine(query_vec, self._matrix)[0].tolist()
            ranked = sorted(
                enumerate(sims), key=lambda x: x[1], reverse=True
            )[:k]
            results: list[L2Match] = []
            for idx, sim in ranked:
                if sim >= self._threshold:
                    results.append(L2Match(
                        template_key=self._kb[idx].key,
                        description=self._kb[idx].description,
                        similarity=round(sim, 4),
                    ))
            if results:
                self._l2_hits += 1
            else:
                self._l2_misses += 1
            return results
        except Exception as exc:
            logger.warning("[L2Semantic] Similarity computation failed: %s", exc)
            self._l2_misses += 1
            return []

    def add_entry(
        self,
        key: str,
        description: str,
        pattern_hint: str = "",
    ) -> None:
        """
        Add a new knowledge-base entry and re-fit the vectorizer.

        Called by the L3 learning loop when a novel pattern is promoted.
        """
        # Avoid duplicates
        existing_keys = {e.key for e in self._kb}
        if key not in existing_keys:
            self._kb.append(_KBEntry(key=key, description=description, pattern_hint=pattern_hint))
            if _HAS_SKLEARN:
                self.fit()
            logger.info("[L2Semantic] Added entry: %s", key)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {value}")
        self._threshold = value

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def knowledge_base_size(self) -> int:
        return len(self._kb)

    @property
    def stats(self) -> dict:
        total = self._l2_hits + self._l2_misses
        return {
            "l2_hits": self._l2_hits,
            "l2_misses": self._l2_misses,
            "l2_hit_rate": round(self._l2_hits / max(1, total), 4),
            "knowledge_base_size": len(self._kb),
            "threshold": self._threshold,
            "fitted": self._fitted,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc(entry: _KBEntry) -> str:
        """Build the TF-IDF document for a knowledge-base entry."""
        parts = [entry.description]
        if entry.pattern_hint:
            # Strip regex metacharacters — raw terms improve recall
            cleaned = _REGEX_NOISE.sub(" ", entry.pattern_hint)
            parts.append(cleaned)
        return " ".join(parts)

    @staticmethod
    def _default_kb() -> list[_KBEntry]:
        """
        Build the default knowledge base from ``l1_cache.BUILTIN_TEMPLATES``.

        Import is deferred here to avoid circular imports at module load.
        """
        from airs_v2.perception.l1_cache import BUILTIN_TEMPLATES
        return [
            _KBEntry(key=key, description=desc, pattern_hint=pattern)
            for key, pattern, desc in BUILTIN_TEMPLATES
        ]
