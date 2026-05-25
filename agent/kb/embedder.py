"""
agent/kb/embedder.py

HuggingFace InferenceClient wrapper for sentence-transformer embeddings.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` via the HF Inference API
(feature_extraction endpoint) to produce 384-dimensional float vectors that
are stored in and queried from Qdrant.

Architecture note
-----------------
``sentence_similarity`` (the API used in the HF docs example) returns
similarity *scores* between pairs of sentences — it does not return an
embedding vector.  To store vectors in Qdrant we need the raw token-level
representations, which come from the ``feature_extraction`` endpoint.
We mean-pool the (tokens × 384) matrix into a single 384-dim sentence vector,
which is the standard approach for all-MiniLM-L6-v2.

The InferenceClient is instantiated once at module load time (via
``_get_client()``) to reuse the underlying HTTP connection pool across many
embedding calls in the same process.  The blocking HTTP call is dispatched to
a thread-pool executor so it never blocks the asyncio event loop.

Public API
----------
    from agent.kb.embedder import embed_text, embed_for_query

    vector: list[float] = await embed_text("database connection pool exhausted")
    query_vector: list[float] = await embed_for_query("OOMKilled Exit Code 137")
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM: int = 384  # Output dimensionality of all-MiniLM-L6-v2


def _settings():
    """Lazy import to avoid circular import at module load time."""
    from config import settings
    return settings


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    """
    Lazily construct a single InferenceClient instance per process.

    Raises:
        RuntimeError: If HF_ACCESS_KEY is not configured.
    """
    cfg = _settings()
    api_key = cfg.HF_ACCESS_KEY
    if not api_key:
        raise RuntimeError(
            "HF_ACCESS_KEY is not set in .env — cannot embed KB entries. "
            "Add HF_ACCESS_KEY=hf_... to your .env file."
        )
    return InferenceClient(provider="hf-inference", api_key=api_key)


async def embed_text(text: str) -> list[float]:
    """
    Produce a 384-dimensional embedding vector for *text*.

    Uses the ``feature_extraction`` endpoint of
    ``sentence-transformers/all-MiniLM-L6-v2`` hosted on HF Inference API.
    The returned (tokens × 384) tensor is mean-pooled into a single
    sentence vector.

    The blocking HTTP call runs in a thread-pool executor so it does not
    block the asyncio event loop during graph node execution.

    Args:
        text: Any plain-text string (alert title, log line, root cause
              narrative, etc.).  Long texts are automatically truncated
              to the model's 256-token context window by the HF backend.

    Returns:
        list[float] of length 384.

    Raises:
        RuntimeError: If HF_ACCESS_KEY is missing.
        Exception:    Re-raises any HF API error so the caller can handle it.
    """
    client = _get_client()

    def _sync_embed() -> list[float]:
        # feature_extraction returns a numpy array or nested list of shape
        # (num_tokens, 384) for all-MiniLM-L6-v2.
        result = client.feature_extraction(text, model=_MODEL)

        # Handle numpy array
        if hasattr(result, "mean"):
            if result.ndim == 2:                    # (tokens, 384) → mean-pool
                vector = result.mean(axis=0).tolist()
            elif result.ndim == 1:                  # already (384,)
                vector = result.tolist()
            else:
                # Unexpected shape — flatten and take first 384 dims
                vector = result.flatten()[:VECTOR_DIM].tolist()
        else:
            # Nested list fallback
            if isinstance(result[0], list):         # [[...], [...], ...]
                import statistics
                vector = [
                    statistics.mean(row[i] for row in result)
                    for i in range(len(result[0]))
                ]
            else:
                vector = list(result)               # already flat

        # Pad or truncate defensively to guarantee exactly VECTOR_DIM elements
        if len(vector) < VECTOR_DIM:
            vector = vector + [0.0] * (VECTOR_DIM - len(vector))
        elif len(vector) > VECTOR_DIM:
            vector = vector[:VECTOR_DIM]

        return vector

    loop = asyncio.get_event_loop()
    vector = await loop.run_in_executor(None, _sync_embed)

    logger.debug(
        "[embedder] embed_text: %d chars → vector dim=%d", len(text), len(vector)
    )
    return vector


async def embed_for_query(query_text: str) -> list[float]:
    """
    Embed a query string for Qdrant nearest-neighbour search.

    For symmetric models like all-MiniLM-L6-v2 the query and document
    embeddings use the same space, so this is identical to ``embed_text``.
    Separated as a distinct function to allow future swap to an asymmetric
    model (e.g. ``multi-qa-MiniLM-L6-cos-v1``) without touching call sites.

    Args:
        query_text: The combined incident alert text used as the search query.

    Returns:
        list[float] of length 384.
    """
    return await embed_text(query_text)
