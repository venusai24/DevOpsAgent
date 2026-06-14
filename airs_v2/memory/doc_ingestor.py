"""
airs_v2/memory/doc_ingestor.py
================================

Curated documentation ingestion pipeline.

Implements Contextual Parent-Child chunking:
  1. Parse a document into sections (parent chunks, ~1500 tokens / ~6000 chars)
  2. Split each section into child chunks (~300 tokens / ~1200 chars)
  3. Generate a deterministic contextual prefix for each child chunk
  4. Embed child chunks (with prefix) for high-precision retrieval
  5. Store parent text as metadata for LLM context expansion on hit

The Corpus/ directory structure is:
  Corpus/
    Diagnosis/     → document_type="troubleshooting"
    Remediation/   → document_type="runbook"

Amendment #5 integration:
  - source_repository field is set from the directory path
  - last_validated is set to ingestion time
  - deprecated defaults to False (set manually for outdated docs)
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Token budget heuristics: 1 token ≈ 4 chars (English technical text)
_PARENT_CHAR_LIMIT = 6000   # ~1500 tokens
_CHILD_CHAR_LIMIT = 1200    # ~300 tokens
_CHILD_OVERLAP = 200        # ~50 tokens of overlap to preserve continuity

# Regex for markdown section headers (H1–H3)
_SECTION_HEADER_RE = re.compile(r"^(#{1,3})\s+.+", re.MULTILINE)

# Known action kinds from ActionKind enum — used for metadata extraction
_ACTION_KIND_KEYWORDS = {
    "restart_pod": ["restart", "pod", "kubectl rollout restart"],
    "scale_deployment": ["scale", "replica", "kubectl scale", "hpa"],
    "rollback_deployment": ["rollback", "revert", "undo", "argocd rollback"],
    "drain_node": ["drain", "cordon", "kubectl drain"],
}


class DocumentIngestor:
    """
    Ingests curated system documentation into the vector store using
    contextual parent-child chunking.

    Parameters
    ----------
    vector_store : VectorStore instance to receive the document chunks.
    """

    def __init__(self, vector_store) -> None:  # vector_store: VectorStore
        self._store = vector_store

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest_document(
        self,
        text: str,
        document_id: str,
        document_title: str,
        document_type: str = "runbook",
        version: str = "1.0.0",
        source_repository: str = "",
    ) -> list:  # list[DocumentChunk]
        """
        Ingest a single document using parent-child chunking.

        Returns the list of DocumentChunk objects that were created and upserted.
        """
        from airs_v2.memory.types import DocumentChunk

        parents = self._split_into_parents(text)
        all_chunks: list[DocumentChunk] = []

        global_chunk_index = 0

        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            section_title = self._extract_section_title(parent_text)
            children = self._split_into_children(parent_text)

            for child_text in children:
                if not child_text.strip():
                    continue

                prefix = self._generate_contextual_prefix(
                    document_title=document_title,
                    section_title=section_title,
                    document_type=document_type,
                    child_text=child_text,
                )

                chunk = DocumentChunk(
                    document_id=document_id,
                    document_title=document_title,
                    document_type=document_type,  # type: ignore[arg-type]
                    section_title=section_title,
                    chunk_text=child_text,
                    contextual_prefix=prefix,
                    parent_chunk_text=parent_text,
                    parent_chunk_id=parent_id,
                    chunk_index=global_chunk_index,
                    total_chunks=0,  # Updated after all chunks are built
                    services_mentioned=self._extract_services(child_text),
                    action_kinds_mentioned=self._extract_action_kinds(child_text),
                    version=version,
                    source_repository=source_repository,
                )
                all_chunks.append(chunk)
                global_chunk_index += 1

        # Update total_chunks now that we know the full count
        for chunk in all_chunks:
            chunk.total_chunks = len(all_chunks)

        if all_chunks:
            self._store.upsert_document_chunks_batch(all_chunks)
            logger.info(
                "[DocumentIngestor] Ingested '%s' → %d chunks.",
                document_title,
                len(all_chunks),
            )

        return all_chunks

    def ingest_corpus_directory(
        self,
        corpus_root: Optional[str | Path] = None,
    ) -> int:
        """
        Ingest all Markdown files from the Corpus/ directory structure.

        Directory → document_type mapping:
          Corpus/Diagnosis/    → "troubleshooting"
          Corpus/Remediation/  → "runbook"

        Returns total number of chunks ingested.
        """
        from config import settings

        root = Path(corpus_root or settings.CORPUS_DIR)
        if not root.exists():
            logger.warning("[DocumentIngestor] Corpus directory not found: %s", root)
            return 0

        type_map: dict[str, str] = {
            "diagnosis": "troubleshooting",
            "remediation": "runbook",
        }

        total_chunks = 0
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            doc_type = type_map.get(subdir.name.lower(), "runbook")

            for md_file in sorted(subdir.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                document_title = md_file.stem.replace("_", " ")
                chunks = self.ingest_document(
                    text=text,
                    document_id=md_file.stem,
                    document_title=document_title,
                    document_type=doc_type,
                    version="1.0.0",
                    source_repository=str(subdir.relative_to(root.parent)),
                )
                total_chunks += len(chunks)

        logger.info(
            "[DocumentIngestor] Corpus ingestion complete: %d total chunks.", total_chunks
        )
        return total_chunks

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _split_into_parents(self, text: str) -> list[str]:
        """
        Split a document into parent chunks at semantic boundaries (headers).

        Strategy: Split at H1/H2/H3 markdown headers. If a section is larger
        than _PARENT_CHAR_LIMIT, further split by double newlines (paragraphs).
        """
        # Find all header positions
        header_positions = [m.start() for m in _SECTION_HEADER_RE.finditer(text)]

        if not header_positions:
            # No headers: split by character limit on paragraph boundaries
            return self._split_by_paragraphs(text, _PARENT_CHAR_LIMIT)

        parents: list[str] = []
        for i, start in enumerate(header_positions):
            end = header_positions[i + 1] if i + 1 < len(header_positions) else len(text)
            section = text[start:end].strip()
            if not section:
                continue
            if len(section) <= _PARENT_CHAR_LIMIT:
                parents.append(section)
            else:
                # Section too large: further split by paragraphs
                parents.extend(self._split_by_paragraphs(section, _PARENT_CHAR_LIMIT))

        # Any text before the first header
        if header_positions[0] > 0:
            preamble = text[: header_positions[0]].strip()
            if preamble:
                parents.insert(0, preamble)

        return parents

    def _split_into_children(self, parent: str) -> list[str]:
        """
        Split a parent chunk into smaller child chunks with overlap.

        Splits on paragraph boundaries (double newlines) to preserve
        semantic coherence within each child chunk.
        """
        if len(parent) <= _CHILD_CHAR_LIMIT:
            return [parent]

        paragraphs = re.split(r"\n\n+", parent)
        children: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > _CHILD_CHAR_LIMIT and current:
                children.append("\n\n".join(current))
                # Overlap: keep last paragraph in the next chunk
                if current:
                    overlap_para = current[-1]
                    current = [overlap_para, para]
                    current_len = len(overlap_para) + para_len
                else:
                    current = [para]
                    current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            children.append("\n\n".join(current))

        return children

    @staticmethod
    def _split_by_paragraphs(text: str, char_limit: int) -> list[str]:
        """Generic paragraph-based splitting with character limit."""
        paragraphs = re.split(r"\n\n+", text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > char_limit and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = [para]
                current_len = len(para)
            else:
                current_parts.append(para)
                current_len += len(para)

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    # ── Prefix generation ──────────────────────────────────────────────────────

    @staticmethod
    def _generate_contextual_prefix(
        document_title: str,
        section_title: str,
        document_type: str,
        child_text: str,
    ) -> str:
        """
        Generate a deterministic contextual prefix for a child chunk.

        This Anthropic-style prefix situates the chunk within its document,
        dramatically improving retrieval precision for small child chunks.
        No LLM call is required — the prefix is constructed from document metadata.
        """
        type_label = {
            "runbook": "SRE Runbook",
            "troubleshooting": "Troubleshooting Guide",
            "sop": "Standard Operating Procedure",
            "architecture": "Architecture Documentation",
            "dependency": "Service Dependency Documentation",
        }.get(document_type, "Technical Document")

        prefix_parts = [
            f"This excerpt is from the {type_label} titled '{document_title}'."
        ]
        if section_title:
            prefix_parts.append(
                f"It is located in the '{section_title}' section."
            )

        # Hint at content type based on keywords
        lower = child_text.lower()
        if any(kw in lower for kw in ["rule", "constraint", "forbidden", "do not"]):
            prefix_parts.append(
                "This section describes a constraint rule or safety boundary."
            )
        elif any(kw in lower for kw in ["kubectl", "command template", "action"]):
            prefix_parts.append(
                "This section contains remediation command templates for an SRE operator."
            )
        elif any(kw in lower for kw in ["symptom", "signature", "indicator", "log"]):
            prefix_parts.append(
                "This section describes diagnostic symptoms or telemetry signatures."
            )

        return " ".join(prefix_parts)

    # ── Metadata extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_section_title(text: str) -> str:
        """Extract the first markdown header from the text as section title."""
        match = _SECTION_HEADER_RE.search(text)
        if match:
            return match.group(0).lstrip("#").strip()
        return ""

    @staticmethod
    def _extract_services(text: str) -> list[str]:
        """
        Extract service names mentioned in the text.

        Looks for common SRE service name patterns: kebab-case names with
        common suffixes like -service, -api, -db, -cache, -worker, -gateway.
        """
        pattern = re.compile(
            r"\b([a-z][a-z0-9-]*(?:service|api|db|cache|worker|gateway|proxy|queue|"
            r"broker|server|node|agent|controller|scheduler|monitor))\b"
        )
        matches = pattern.findall(text.lower())
        # Deduplicate and return up to 10 matches to keep metadata manageable
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result[:10]

    @staticmethod
    def _extract_action_kinds(text: str) -> list[str]:
        """Extract ActionKind references from text based on keyword matching."""
        found: list[str] = []
        lower = text.lower()
        for kind, keywords in _ACTION_KIND_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                found.append(kind)
        return found
