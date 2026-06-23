"""
Embedding Service — Module 1.8.

Provides dense embeddings using BAAI/bge-large-en-v1.5 (768-dim).

Phase 1: HuggingFace Inference API (remote).
Phase 2: Local model inference via sentence-transformers (when GPU available).

The interface is identical in both phases — only the backend changes.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

_ENCODER = None  # Module-level cached encoder


def get_local_encoder():
    """
    Lazy-load the sentence-transformers encoder.
    Cached after first call so model is only loaded once per worker process.
    """
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading BAAI/bge-large-en-v1.5 encoder (first call — may take a moment)")
        _ENCODER = SentenceTransformer("BAAI/bge-large-en-v1.5")
        log.info("Encoder loaded. Embedding dim: 768")
    return _ENCODER


class EmbeddingService:
    """
    Single-responsibility embedding service.

    Tries local sentence-transformers first (faster, no API cost).
    Falls back to HuggingFace Inference API if local model unavailable.
    """

    def __init__(
        self,
        hf_key: str = "",
        prefer_local: bool = True,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._hf_key = hf_key
        self._prefer_local = prefer_local
        self._timeout = timeout
        self._max_retries = max_retries
        self._HF_URL = (
            "https://api-inference.huggingface.co/pipeline/feature-extraction"
            "/BAAI/bge-large-en-v1.5"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into 768-dim vectors.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of 768-dim float vectors.

        Raises:
            RuntimeError: If all embedding backends fail.
        """
        if not texts:
            return []

        if self._prefer_local:
            try:
                return self._embed_local(texts)
            except Exception as e:
                log.warning("Local embedding failed (%s) — falling back to HF API", e)

        return self._embed_hf_api(texts)

    def embed_single(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper."""
        return self.embed([text])[0]

    # ─── Backends ────────────────────────────────────────────────────────────

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Local inference via sentence-transformers."""
        encoder = get_local_encoder()
        embeddings = encoder.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in embeddings]

    def _embed_hf_api(self, texts: list[str]) -> list[list[float]]:
        """HuggingFace Inference API fallback."""
        import httpx

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = httpx.post(
                    self._HF_URL,
                    headers={"Authorization": f"Bearer {self._hf_key}"},
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                raw = resp.json()

                # Shape: [batch, dim] or [batch, seq, dim]
                if isinstance(raw[0][0], list):
                    # Mean pool over sequence length
                    pooled = []
                    for seq in raw:
                        n = len(seq)
                        dim = len(seq[0])
                        vec = [sum(seq[t][d] for t in range(n)) / n for d in range(dim)]
                        pooled.append(vec)
                    return pooled

                return raw  # Already [batch, dim]

            except Exception as e:
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    log.warning("HF API attempt %d failed: %s — retrying in %ds", attempt, e, wait)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"HF API embedding failed after {attempt} attempts: {e}") from e

        raise RuntimeError("All embedding attempts exhausted")


# ─── Module-level singleton ───────────────────────────────────────────────────

_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Return the global EmbeddingService singleton."""
    global _service
    if _service is None:
        from airs.config import settings
        _service = EmbeddingService(
            hf_key=settings.hf_key,
            prefer_local=True,
        )
    return _service
