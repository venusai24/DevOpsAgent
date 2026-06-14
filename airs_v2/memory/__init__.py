"""
airs_v2.memory — RAG Memory Subsystem (Stage 5).

Modules
-------
types           — Pydantic models: PostMortemTrace, HumanFeedback, DocumentChunk,
                  RAGQuery, RAGResult, RAGResponse, IncidentOutcome.
embedding_service — Singleton embedding service (BGE-large/GPU or MiniLM/CPU).
vector_store    — ChromaDB dual-collection store (incidents + documentation).
reranker        — Cross-encoder reranking (configurable model via config.py).
graph_store     — GraphStore protocol + NetworkXGraphStore with JSON persistence.
doc_ingestor    — Contextual parent-child chunking for the Corpus/ directory.
rag_engine      — Hierarchical RAG orchestrator (retrieval + graph enrichment).
learning_loop   — Post-resolution ingestion + PII redaction + feedback knowledge.
"""

from airs_v2.memory.types import (
    HumanFeedback,
    IncidentOutcome,
    PostMortemTrace,
    DocumentChunk,
    RAGQuery,
    RAGResult,
    RAGResponse,
)
from airs_v2.memory.embedding_service import EmbeddingService
from airs_v2.memory.vector_store import VectorStore
from airs_v2.memory.reranker import CrossEncoderReranker
from airs_v2.memory.graph_store import GraphStore, NetworkXGraphStore
from airs_v2.memory.doc_ingestor import DocumentIngestor
from airs_v2.memory.rag_engine import RAGEngine
from airs_v2.memory.learning_loop import LearningLoop, _redact_sensitive_data

__all__ = [
    "HumanFeedback",
    "IncidentOutcome",
    "PostMortemTrace",
    "DocumentChunk",
    "RAGQuery",
    "RAGResult",
    "RAGResponse",
    "EmbeddingService",
    "VectorStore",
    "CrossEncoderReranker",
    "GraphStore",
    "NetworkXGraphStore",
    "DocumentIngestor",
    "RAGEngine",
    "LearningLoop",
    "_redact_sensitive_data",
]
