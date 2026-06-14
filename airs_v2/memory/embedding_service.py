"""
airs_v2/memory/embedding_service.py
=====================================

Singleton embedding service with automatic GPU/CPU model selection.

Model selection strategy
------------------------
GPU available → BAAI/bge-large-en-v1.5   (1024-dim, MTEB ~63.5, gated via HF_KEY)
GPU absent    → sentence-transformers/all-MiniLM-L6-v2 (384-dim, MTEB ~49.5, CPU)

BGE instruction prefix
-----------------------
BGE-large is an instruction-tuned model. Queries must be prefixed with:
  "Represent this sentence for retrieval: "
Documents are embedded without a prefix (asymmetric retrieval).

HF_KEY authentication
-----------------------
BAAI/bge-large-en-v1.5 is a gated model on HuggingFace. The HF_KEY from
.env is loaded and exported as HF_TOKEN before the model is downloaded.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_BGE_MODEL = "BAAI/bge-large-en-v1.5"
_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_BGE_QUERY_PREFIX = "Represent this sentence for retrieval: "


class EmbeddingService:
    """
    Thread-safe singleton embedding service.

    Usage
    -----
    service = EmbeddingService.get_instance()
    vectors = service.embed_batch(["some text", "another text"])
    query_vec = service.embed_query("connection pool exhausted")
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or self._auto_select_model()
        self._model = None          # Lazy-loaded on first use
        self._is_bge = "bge" in self._model_name.lower()
        self._setup_hf_token()
        logger.info(
            "[EmbeddingService] Initialised model=%s is_bge=%s",
            self._model_name,
            self._is_bge,
        )

    # ── Singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls, model_name: Optional[str] = None
    ) -> "EmbeddingService":
        """Return the process-level singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful in tests)."""
        cls._instance = None

    # ── Model selection ────────────────────────────────────────────────────────

    @staticmethod
    def _auto_select_model() -> str:
        """Select the best embedding model based on available hardware."""
        try:
            import torch  # noqa: F401 — may not be installed
            if torch.cuda.is_available():
                logger.info(
                    "[EmbeddingService] GPU detected → selecting %s", _BGE_MODEL
                )
                return _BGE_MODEL
        except ImportError:
            pass
        logger.info(
            "[EmbeddingService] No GPU → selecting %s", _MINILM_MODEL
        )
        return _MINILM_MODEL

    @staticmethod
    def _setup_hf_token() -> None:
        """
        Load HF_KEY from environment / .env and export as HF_TOKEN.

        This is required for gated model downloads (BGE-large is gated).
        python-dotenv is already a project dependency so .env is loaded
        by pydantic-settings when the app starts.
        """
        hf_key = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
        if hf_key:
            os.environ["HF_TOKEN"] = hf_key
            logger.debug("[EmbeddingService] HF_TOKEN set from HF_KEY")
        else:
            logger.debug("[EmbeddingService] No HF_KEY found; public models only")

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # noqa: F401

        logger.info("[EmbeddingService] Loading model %s …", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("[EmbeddingService] Model loaded.")

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the active model."""
        return 1024 if self._is_bge else 384

    def embed(self, text: str) -> list[float]:
        """Embed a single document string (no instruction prefix)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings. Returns list of float vectors."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a retrieval query string.

        For BGE models: prepends the instruction prefix for asymmetric retrieval.
        For other models: embeds the text directly.
        """
        if self._is_bge:
            text = f"{_BGE_QUERY_PREFIX}{text}"
        return self.embed(text)
