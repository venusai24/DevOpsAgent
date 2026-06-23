"""
Document Ingestion Pipeline — Module 1.5.

Decomposes existing Markdown corpus documents into atomic YAML-aligned
documents and vectorizes them into the 6 Qdrant collections.

Design decisions:
  - Each Markdown section (## header) → potentially multiple atomic documents
  - Each atomic document = ONE independently retrievable knowledge unit
  - Embedded in BAAI/bge-large-en-v1.5 (768-dim) via HuggingFace Inference API
  - Upserted with structured metadata payload for pre-filtering in ConANN

Corpus → Collection mapping:
  Corpus/Diagnosis/*.md   → C1: diagnostic_knowledge
  Corpus/Remediation/*.md → C4: remediation_actions + C6: operational_constraints
  knowledge/**/*.yaml     → respective collection per YAML metadata
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

log = logging.getLogger(__name__)


# ─── Collection Names ────────────────────────────────────────────────────────

class Collection(str, Enum):
    DIAGNOSTIC_KNOWLEDGE = "diagnostic_knowledge"
    TOOL_SELECTION = "tool_selection"
    QUERY_TEMPLATES = "query_templates"
    REMEDIATION_ACTIONS = "remediation_actions"
    FAILURE_SIGNATURES = "failure_signatures"
    OPERATIONAL_CONSTRAINTS = "operational_constraints"

    @classmethod
    def all_names(cls) -> list[str]:
        return [c.value for c in cls]


# ─── Atomic Document ─────────────────────────────────────────────────────────

@dataclass
class AtomicDocument:
    """
    A single, independently retrievable knowledge unit.

    One AtomicDocument → one Qdrant point with a 768-dim embedding.
    """
    document_id: str
    collection: str
    knowledge_type: str              # DECLARATIVE | PROCEDURAL | HEURISTIC | NORMATIVE
    content: str                     # The text to embed (serialised content block)
    title: str = ""
    source_file: str = ""
    # Metadata for Qdrant payload pre-filtering
    trigger_state: Optional[str] = None          # Continue | Diagnose | Abstain_Prune | Escalate
    signal_types: list[str] = field(default_factory=list)  # METRICS | LOGS | TRACES | ...
    failure_domain: Optional[str] = None         # data_tier | network | cluster | application
    fault_categories: list[str] = field(default_factory=list)
    applicable_tools: list[str] = field(default_factory=list)
    tool_tier: Optional[int] = None              # 1 | 2 | 3
    max_step_risk: Optional[float] = None
    requires_idempotency: bool = False
    differential_diagnosis: list[str] = field(default_factory=list)  # related doc IDs


# ─── Embedding Service ───────────────────────────────────────────────────────

class EmbeddingService:
    """
    Wraps the HuggingFace Inference API for BAAI/bge-large-en-v1.5 embeddings.
    Includes retry logic and batching.
    """
    HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    MODEL = "BAAI/bge-large-en-v1.5"

    def __init__(self, hf_key: str, timeout: int = 60, max_retries: int = 3) -> None:
        self._hf_key = hf_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._url = self.HF_API_URL.format(model=self.MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 768-dim float vectors (one per input text).

        Raises:
            RuntimeError: If embedding fails after all retries.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                response = httpx.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._hf_key}"},
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                embeddings = response.json()

                # HF returns either list[list[float]] or list[list[list[float]]]
                # bge-large returns [batch, seq_len, dim] — take CLS (index 0) or mean pool
                if isinstance(embeddings[0][0], list):
                    # Shape: [batch, seq_len, dim] → mean pool along seq_len
                    pooled = []
                    for seq in embeddings:
                        n = len(seq)
                        vec = [sum(seq[t][d] for t in range(n)) / n for d in range(len(seq[0]))]
                        pooled.append(vec)
                    return pooled
                return embeddings  # Already [batch, dim]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and attempt < self._max_retries:
                    wait = 2 ** attempt
                    log.warning("HF API 503 — model loading. Retrying in %ds...", wait)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"HF API error after {attempt} attempts: {e}") from e
            except Exception as e:
                if attempt < self._max_retries:
                    log.warning("Embedding attempt %d failed: %s — retrying", attempt, e)
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Embedding failed: {e}") from e

        raise RuntimeError("Embedding failed after all retries")


# ─── Markdown Decomposer ──────────────────────────────────────────────────────

class MarkdownDecomposer:
    """
    Decomposes a Markdown corpus file into atomic documents.

    Strategy:
    1. Split by `##` (H2) headers — each section is a logical unit.
    2. Within each section, look for enumerated rules (numbered lists,
       bolded terms, IF...THEN patterns) and split further.
    3. Each atomic unit gets metadata extracted from its content.
    """

    # Patterns that indicate a new atomic rule within a section
    _RULE_PATTERNS = [
        re.compile(r'^#{3,4}\s+(.+)$', re.MULTILINE),      # ### or #### header
        re.compile(r'^\*\*Rule\s+[\dA-Z]+[\.\:]\*\*', re.MULTILINE),  # **Rule 1A:**
        re.compile(r'^\d+\.\s+\*\*', re.MULTILINE),        # 1. **Bold item**
    ]

    # Signal type keywords for automatic detection
    _SIGNAL_KEYWORDS = {
        "METRICS": ["metric", "prometheus", "gauge", "counter", "rate", "cpu", "memory", "latency"],
        "LOGS": ["log", "error log", "loki", "opensearch", "stderr", "event log"],
        "TRACES": ["trace", "span", "jaeger", "tempo", "distributed trace"],
        "K8S_STATE": ["kubectl", "pod", "deployment", "namespace", "node", "event"],
        "CODE": ["git", "diff", "commit", "repository", "source"],
    }

    # Failure domain keywords
    _DOMAIN_KEYWORDS = {
        "data_tier": ["database", "cache", "redis", "valkey", "postgres", "mysql", "connection pool", "replica"],
        "network": ["dns", "network", "tcp", "udp", "packet", "latency", "timeout", "firewall", "ingress", "egress"],
        "cluster_management": ["pod", "node", "scheduler", "kubelet", "pvc", "persistent volume", "eviction"],
        "application": ["oom", "crash", "exception", "error rate", "thread", "heap", "gc", "goroutine"],
    }

    def decompose(
        self,
        file_path: Path,
        default_collection: str,
        default_knowledge_type: str,
    ) -> list[AtomicDocument]:
        """
        Decompose a Markdown file into atomic documents.

        Args:
            file_path:              Path to the .md file.
            default_collection:     Collection to assign (e.g., 'diagnostic_knowledge').
            default_knowledge_type: Knowledge type (e.g., 'HEURISTIC').

        Returns:
            List of AtomicDocument instances ready for embedding.
        """
        text = file_path.read_text(encoding="utf-8")
        sections = self._split_by_h2(text, file_path.stem)
        documents: list[AtomicDocument] = []

        for section_title, section_text in sections:
            # Attempt sub-decomposition within the section
            sub_units = self._split_section(section_title, section_text)
            for i, (title, content) in enumerate(sub_units):
                doc_id = self._make_doc_id(file_path.stem, section_title, i)
                signal_types = self._detect_signal_types(content)
                domain = self._detect_domain(content)

                doc = AtomicDocument(
                    document_id=doc_id,
                    collection=default_collection,
                    knowledge_type=default_knowledge_type,
                    content=f"{title}\n\n{content}".strip(),
                    title=title,
                    source_file=file_path.name,
                    signal_types=signal_types,
                    failure_domain=domain,
                )
                documents.append(doc)

        log.info(
            "Decomposed %s → %d atomic documents", file_path.name, len(documents)
        )
        return documents

    def _split_by_h2(
        self, text: str, file_stem: str
    ) -> list[tuple[str, str]]:
        """Split by ## headers. Returns (title, body) pairs."""
        sections: list[tuple[str, str]] = []
        # Find all ## headers and their positions
        pattern = re.compile(r'^##\s+(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            # No H2 headers — treat whole file as one section
            return [(file_stem, text)]

        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((title, body))

        return sections

    def _split_section(
        self, section_title: str, section_text: str
    ) -> list[tuple[str, str]]:
        """
        Attempt to split a section into sub-units by H3/H4 headers or rule patterns.
        Falls back to returning the whole section as one unit.
        """
        # Try H3/H4 split first
        h3_pattern = re.compile(r'^#{3,4}\s+(.+)$', re.MULTILINE)
        matches = list(h3_pattern.finditer(section_text))

        if len(matches) >= 2:
            units: list[tuple[str, str]] = []
            for i, match in enumerate(matches):
                title = f"{section_title} — {match.group(1).strip()}"
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
                body = section_text[start:end].strip()
                if body:
                    units.append((title, body))
            return units if units else [(section_title, section_text)]

        # Return section as-is
        return [(section_title, section_text)]

    def _detect_signal_types(self, text: str) -> list[str]:
        """Detect which observability signal types this text references."""
        text_lower = text.lower()
        found = []
        for signal, keywords in self._SIGNAL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(signal)
        return found or ["METRICS"]  # Default to METRICS if undetectable

    def _detect_domain(self, text: str) -> Optional[str]:
        """Detect the failure domain from text content."""
        text_lower = text.lower()
        scores = {
            domain: sum(1 for kw in keywords if kw in text_lower)
            for domain, keywords in self._DOMAIN_KEYWORDS.items()
        }
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else None

    @staticmethod
    def _make_doc_id(file_stem: str, section_title: str, index: int) -> str:
        """Generate a stable, human-readable document ID."""
        # Sanitise: lowercase, replace spaces/special chars with underscores
        stem = re.sub(r'[^a-z0-9]+', '_', file_stem.lower()).strip('_')
        section = re.sub(r'[^a-z0-9]+', '_', section_title.lower()).strip('_')[:30]
        return f"{stem}__{section}__{index:02d}"


# ─── YAML Document Loader ────────────────────────────────────────────────────

class YAMLDocumentLoader:
    """
    Loads pre-synthesized atomic YAML documents from the knowledge/ directory.
    Each YAML file is already one atomic document — no decomposition needed.
    """

    REQUIRED_FIELDS = {"document_id", "collection", "knowledge_type", "content"}

    def load(self, yaml_path: Path) -> Optional[AtomicDocument]:
        """
        Load a single YAML file as an AtomicDocument.

        Returns None and logs a warning if the file is malformed.
        """
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            log.warning("Failed to parse YAML %s: %s", yaml_path, e)
            return None

        if not isinstance(data, dict):
            log.warning("YAML %s is not a dict — skipping", yaml_path)
            return None

        missing = self.REQUIRED_FIELDS - set(data.keys())
        if missing:
            log.warning("YAML %s missing required fields %s — skipping", yaml_path, missing)
            return None

        content = data["content"]
        if isinstance(content, dict):
            content = yaml.dump(content, default_flow_style=False)
        elif not isinstance(content, str):
            content = str(content)

        return AtomicDocument(
            document_id=data["document_id"],
            collection=data["collection"],
            knowledge_type=data["knowledge_type"],
            content=content,
            title=data.get("title", data["document_id"]),
            source_file=yaml_path.name,
            trigger_state=data.get("trigger_state"),
            signal_types=data.get("signal_types", []),
            failure_domain=data.get("failure_domain"),
            fault_categories=data.get("fault_categories", []),
            applicable_tools=data.get("applicable_tools", []),
            tool_tier=data.get("tool_tier"),
            max_step_risk=data.get("max_step_risk"),
            requires_idempotency=data.get("requires_idempotency", False),
            differential_diagnosis=data.get("differential_diagnosis", []),
        )


# ─── Ingestion Pipeline ──────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline:
    1. Decompose/load documents
    2. Embed via HuggingFace API
    3. Upsert to Qdrant with metadata payload
    """

    BATCH_SIZE = 16  # Texts per HF API call

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_service: EmbeddingService,
    ) -> None:
        self._qdrant = qdrant_client
        self._embedder = embedding_service
        self._md_decomposer = MarkdownDecomposer()
        self._yaml_loader = YAMLDocumentLoader()

    def ingest_documents(
        self,
        documents: list[AtomicDocument],
        verbose: bool = False,
    ) -> dict[str, int]:
        """
        Embed and upsert a list of AtomicDocuments to Qdrant.

        Args:
            documents: List of documents to ingest.
            verbose:   Log each document as it is ingested.

        Returns:
            Dict mapping collection name → number of documents upserted.
        """
        if not documents:
            log.warning("ingest_documents called with empty document list")
            return {}

        counts: dict[str, int] = {}

        # Process in batches
        for batch_start in range(0, len(documents), self.BATCH_SIZE):
            batch = documents[batch_start : batch_start + self.BATCH_SIZE]

            # Step 1: Embed
            texts = [doc.content for doc in batch]
            try:
                embeddings = self._embedder.embed(texts)
            except RuntimeError as e:
                log.error("Embedding batch %d failed: %s — skipping batch", batch_start, e)
                continue

            if len(embeddings) != len(batch):
                log.error(
                    "Embedding count mismatch: expected %d, got %d — skipping batch",
                    len(batch),
                    len(embeddings),
                )
                continue

            # Step 2: Group by collection and upsert
            collection_points: dict[str, list[PointStruct]] = {}
            for doc, embedding in zip(batch, embeddings):
                point = self._make_point(doc, embedding)
                collection_points.setdefault(doc.collection, []).append(point)

            for collection_name, points in collection_points.items():
                try:
                    self._qdrant.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True,
                    )
                    counts[collection_name] = counts.get(collection_name, 0) + len(points)
                    if verbose:
                        for p in points:
                            log.info(
                                "[%s] upserted %s", collection_name, p.payload.get("document_id")
                            )
                except Exception as e:
                    log.error(
                        "Qdrant upsert to %s failed: %s", collection_name, e
                    )

        for col, count in counts.items():
            log.info("Ingested %d documents into collection '%s'", count, col)

        return counts

    def ingest_markdown_directory(
        self,
        directory: Path,
        collection: str,
        knowledge_type: str,
        verbose: bool = False,
    ) -> dict[str, int]:
        """
        Decompose all .md files in a directory and ingest into Qdrant.

        Args:
            directory:      Path to directory containing .md files.
            collection:     Target Qdrant collection name.
            knowledge_type: Knowledge type for all documents in this directory.
            verbose:        Log each document.

        Returns:
            Counts dict from ingest_documents().
        """
        md_files = sorted(directory.glob("*.md"))
        if not md_files:
            log.warning("No .md files found in %s", directory)
            return {}

        all_docs: list[AtomicDocument] = []
        for md_file in md_files:
            docs = self._md_decomposer.decompose(md_file, collection, knowledge_type)
            all_docs.extend(docs)

        log.info(
            "Found %d .md files in %s → %d atomic documents",
            len(md_files), directory, len(all_docs),
        )
        return self.ingest_documents(all_docs, verbose=verbose)

    def ingest_yaml_directory(
        self,
        directory: Path,
        verbose: bool = False,
    ) -> dict[str, int]:
        """
        Load all .yaml files recursively from a knowledge/ subdirectory and ingest.

        Each YAML file is already an atomic document — collection is read
        from the YAML metadata field.
        """
        yaml_files = sorted(directory.rglob("*.yaml"))
        if not yaml_files:
            log.info("No .yaml files found in %s", directory)
            return {}

        all_docs: list[AtomicDocument] = []
        for yaml_file in yaml_files:
            doc = self._yaml_loader.load(yaml_file)
            if doc is not None:
                all_docs.append(doc)

        log.info(
            "Found %d .yaml files in %s → %d valid atomic documents",
            len(yaml_files), directory, len(all_docs),
        )
        return self.ingest_documents(all_docs, verbose=verbose)

    @staticmethod
    def _make_point(doc: AtomicDocument, embedding: list[float]) -> PointStruct:
        """Convert an AtomicDocument + embedding → Qdrant PointStruct."""
        # Deterministic integer ID from document_id string
        point_id = int(
            hashlib.sha256(doc.document_id.encode()).hexdigest()[:15], 16
        )

        payload: dict[str, Any] = {
            "document_id": doc.document_id,
            "collection": doc.collection,
            "knowledge_type": doc.knowledge_type,
            "title": doc.title,
            "source_file": doc.source_file,
            "content_preview": doc.content[:200],  # For debugging
        }

        # Add optional metadata fields (skip None values)
        if doc.trigger_state:
            payload["trigger_state"] = doc.trigger_state
        if doc.signal_types:
            payload["signal_types"] = doc.signal_types
        if doc.failure_domain:
            payload["failure_domain"] = doc.failure_domain
        if doc.fault_categories:
            payload["fault_categories"] = doc.fault_categories
        if doc.applicable_tools:
            payload["applicable_tools"] = doc.applicable_tools
        if doc.tool_tier is not None:
            payload["tool_tier"] = doc.tool_tier
        if doc.max_step_risk is not None:
            payload["max_step_risk"] = doc.max_step_risk
        if doc.requires_idempotency:
            payload["requires_idempotency"] = doc.requires_idempotency
        if doc.differential_diagnosis:
            payload["differential_diagnosis"] = doc.differential_diagnosis

        return PointStruct(id=point_id, vector=embedding, payload=payload)


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_ingestion_pipeline(
    qdrant_url: str,
    hf_key: str,
) -> IngestionPipeline:
    """
    Create an IngestionPipeline from connection parameters.

    Usage:
        pipeline = create_ingestion_pipeline(settings.qdrant_url, settings.hf_key)
    """
    client = QdrantClient(url=qdrant_url)
    embedder = EmbeddingService(hf_key=hf_key)
    return IngestionPipeline(client, embedder)
