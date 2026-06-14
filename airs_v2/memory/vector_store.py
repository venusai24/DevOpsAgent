"""
airs_v2/memory/vector_store.py
================================

ChromaDB-backed dual-collection vector store.

Two collections
---------------
postmortem_traces   — Historical incident records (Domain A).
                      Embedding: PostMortemTrace.retrieval_embedding_text()
                      Metadata : PostMortemTrace.to_vector_metadata()

system_documentation — Curated static documents (Domain B).
                      Embedding: DocumentChunk.to_embedding_text()
                      Metadata : DocumentChunk.to_vector_metadata()

Retrieval strategy
------------------
query_hierarchical() implements the two-phase priority logic:
  Phase 1: Query incidents. If ≥ SUFFICIENCY_COUNT results with
           similarity ≥ SUFFICIENCY_SIMILARITY → return immediately.
  Phase 2: Otherwise, also query documentation and merge results.

Pre-filtering
-------------
All queries apply mandatory metadata filters before the KNN search:
  Incidents  : exclude outcome="rejected" (safety invariant)
               exclude deprecated=True docs (Amendment #5)
  Docs       : exclude deprecated="True" (Amendment #5)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB persistent dual-collection vector store.

    Parameters
    ----------
    persist_directory : Path to ChromaDB persistence directory.
    embedding_service : EmbeddingService instance (auto-created if None).
    """

    INCIDENTS_COLLECTION = "postmortem_traces"
    DOCS_COLLECTION = "system_documentation"

    def __init__(
        self,
        persist_directory: Optional[str | Path] = None,
        embedding_service=None,
    ) -> None:
        import chromadb
        from config import settings
        from airs_v2.memory.embedding_service import EmbeddingService

        persist_dir = str(persist_directory or settings.CHROMA_PERSIST_DIR)
        self._embedding = embedding_service or EmbeddingService.get_instance()
        self._client = chromadb.PersistentClient(path=persist_dir)

        self._incidents = self._client.get_or_create_collection(
            name=self.INCIDENTS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._docs = self._client.get_or_create_collection(
            name=self.DOCS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "[VectorStore] Ready. incidents=%d docs=%d",
            self._incidents.count(),
            self._docs.count(),
        )

    # ── Write: Incidents ───────────────────────────────────────────────────────

    def upsert_trace(self, trace) -> None:  # trace: PostMortemTrace
        """Upsert a single PostMortemTrace into the incidents collection."""
        embedding_text = trace.retrieval_embedding_text()
        embedding = self._embedding.embed(embedding_text)
        self._incidents.upsert(
            ids=[trace.trace_id],
            embeddings=[embedding],
            documents=[embedding_text],
            metadatas=[trace.to_vector_metadata()],
        )
        logger.debug("[VectorStore] Upserted trace %s", trace.trace_id)

    def upsert_traces_batch(self, traces: list) -> None:
        """Batch upsert a list of PostMortemTrace objects."""
        if not traces:
            return
        texts = [t.retrieval_embedding_text() for t in traces]
        embeddings = self._embedding.embed_batch(texts)
        self._incidents.upsert(
            ids=[t.trace_id for t in traces],
            embeddings=embeddings,
            documents=texts,
            metadatas=[t.to_vector_metadata() for t in traces],
        )
        logger.info("[VectorStore] Batch upserted %d traces.", len(traces))

    # ── Write: Documentation ───────────────────────────────────────────────────

    def upsert_document_chunk(self, chunk) -> None:  # chunk: DocumentChunk
        """Upsert a single DocumentChunk into the documentation collection."""
        embedding_text = chunk.to_embedding_text()
        embedding = self._embedding.embed(embedding_text)
        self._docs.upsert(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[embedding_text],
            metadatas=[chunk.to_vector_metadata()],
        )
        logger.debug("[VectorStore] Upserted doc chunk %s", chunk.chunk_id)

    def upsert_document_chunks_batch(self, chunks: list) -> None:
        """Batch upsert a list of DocumentChunk objects."""
        if not chunks:
            return
        texts = [c.to_embedding_text() for c in chunks]
        embeddings = self._embedding.embed_batch(texts)
        self._docs.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.to_vector_metadata() for c in chunks],
        )
        logger.info("[VectorStore] Batch upserted %d doc chunks.", len(chunks))

    # ── Read: Incidents ────────────────────────────────────────────────────────

    def query_incidents(self, query) -> list:  # query: RAGQuery → list[RAGResult]
        """
        Query the incidents collection with metadata pre-filtering.

        Safety filters applied:
          - exclude outcome="rejected"   (prevents bad examples from contaminating retrieval)
          - filter by focal_service if specified (domain isolation)
        """
        from airs_v2.memory.types import RAGResult, PostMortemTrace, IncidentOutcome

        query_vec = self._embedding.embed_query(query.query_text)

        where_clause: dict = {}
        if query.exclude_rejected:
            where_clause["outcome"] = {"$ne": IncidentOutcome.REJECTED.value}

        t0 = time.perf_counter()
        chroma_results = self._incidents.query(
            query_embeddings=[query_vec],
            n_results=min(query.candidate_k, max(1, self._incidents.count())),
            where=where_clause if where_clause else None,
            include=["metadatas", "documents", "distances"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        results: list[RAGResult] = []
        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return results

        for i, (doc_id, distance, metadata) in enumerate(zip(
            chroma_results["ids"][0],
            chroma_results["distances"][0],
            chroma_results["metadatas"][0],
        )):
            # ChromaDB cosine distance → similarity: 1 - distance
            similarity = max(0.0, 1.0 - distance)
            if similarity < query.min_similarity:
                continue

            # Reconstruct a lightweight trace shell for the result
            trace = PostMortemTrace(
                trace_id=metadata.get("trace_id", doc_id),
                incident_id=metadata.get("incident_id", ""),
                focal_service=metadata.get("focal_service", ""),
                incident_type=metadata.get("incident_type", ""),
                root_cause_node=metadata.get("root_cause_node", ""),
                outcome=IncidentOutcome(
                    metadata.get("outcome", IncidentOutcome.APPROVED.value)
                ),
                diagnosis_confidence=float(metadata.get("diagnosis_confidence", 0.0)),
                action_confidence=float(metadata.get("action_confidence", 0.0)),
                symbolic_path=metadata.get("symbolic_path", "SYMBOLIC_FAST"),
                resolution_duration_s=float(metadata.get("resolution_duration_s", 0.0)),
                blast_radius_depth=int(metadata.get("blast_radius_depth", 0)),
                created_at=metadata.get("created_at", ""),
            )
            results.append(
                RAGResult(
                    trace=trace,
                    source="incident",
                    similarity_score=similarity,
                    retrieval_rank=i + 1,
                )
            )

        logger.debug(
            "[VectorStore] Incident query returned %d results in %.1fms",
            len(results),
            latency_ms,
        )
        return results

    # ── Read: Documentation ────────────────────────────────────────────────────

    def query_documentation(self, query) -> list:  # query: RAGQuery → list[RAGResult]
        """
        Query the documentation collection with metadata pre-filtering.

        Amendment #5: excludes deprecated="True" docs automatically.
        """
        from airs_v2.memory.types import RAGResult, DocumentChunk

        query_vec = self._embedding.embed_query(query.query_text)

        where_clause: dict = {}
        if query.exclude_deprecated_docs:
            where_clause["deprecated"] = {"$eq": "False"}

        t0 = time.perf_counter()
        chroma_results = self._docs.query(
            query_embeddings=[query_vec],
            n_results=min(query.candidate_k, max(1, self._docs.count())),
            where=where_clause if where_clause else None,
            include=["metadatas", "documents", "distances"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        results: list[RAGResult] = []
        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return results

        for i, (doc_id, distance, document, metadata) in enumerate(zip(
            chroma_results["ids"][0],
            chroma_results["distances"][0],
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
        )):
            similarity = max(0.0, 1.0 - distance)
            if similarity < query.min_similarity:
                continue

            chunk = DocumentChunk(
                chunk_id=metadata.get("chunk_id", doc_id),
                document_id=metadata.get("document_id", ""),
                document_title=metadata.get("document_title", ""),
                document_type=metadata.get("document_type", "runbook"),  # type: ignore[arg-type]
                section_title=metadata.get("section_title", ""),
                chunk_text=document,
                contextual_prefix="",
                parent_chunk_text="",  # Stored separately; fetched on demand
                parent_chunk_id=metadata.get("parent_chunk_id", ""),
                chunk_index=int(metadata.get("chunk_index", 0)),
                total_chunks=int(metadata.get("total_chunks", 0)),
                services_mentioned=metadata.get("services_mentioned", "").split(",")
                if metadata.get("services_mentioned")
                else [],
                source_repository=metadata.get("source_repository", ""),
                deprecated=metadata.get("deprecated", "False") == "True",
            )
            results.append(
                RAGResult(
                    document_chunk=chunk,
                    source="documentation",
                    similarity_score=similarity,
                    retrieval_rank=i + 1,
                )
            )

        logger.debug(
            "[VectorStore] Doc query returned %d results in %.1fms",
            len(results),
            latency_ms,
        )
        return results

    # ── Hierarchical retrieval ─────────────────────────────────────────────────

    def query_hierarchical(
        self, query
    ) -> tuple[list, bool]:  # (list[RAGResult], fallback_triggered)
        """
        Two-phase hierarchical retrieval.

        Phase 1: Query incidents. If sufficiency threshold is met, return.
        Phase 2: If insufficient, also query documentation and merge results.

        Returns
        -------
        (merged_results, fallback_triggered)
          fallback_triggered=True means documentation was also queried.
        """
        from config import settings

        incident_results = self.query_incidents(query)

        # Check sufficiency: ≥ N approved results with similarity ≥ threshold
        high_quality = [
            r for r in incident_results
            if r.similarity_score >= settings.RAG_INCIDENT_SUFFICIENCY_SIMILARITY
        ]
        if len(high_quality) >= settings.RAG_INCIDENT_SUFFICIENCY_COUNT:
            logger.debug(
                "[VectorStore] Incident sufficiency met (%d results) — skipping docs.",
                len(high_quality),
            )
            return incident_results, False

        # Phase 2: also query documentation
        if query.include_documentation and self._docs.count() > 0:
            doc_results = self.query_documentation(query)
            merged = incident_results + doc_results
            logger.debug(
                "[VectorStore] Hierarchical fallback: %d incidents + %d docs.",
                len(incident_results),
                len(doc_results),
            )
            return merged, True

        return incident_results, False

    # ── Utility ────────────────────────────────────────────────────────────────

    def incident_count(self) -> int:
        return self._incidents.count()

    def doc_count(self) -> int:
        return self._docs.count()
