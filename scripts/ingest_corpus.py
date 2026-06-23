#!/usr/bin/env python3
"""
ingest_corpus.py — Corpus Ingestion Script.

Decomposes all existing Corpus/ Markdown files and synthesized knowledge/
YAML files into atomic documents, embeds them, and upserts to the 6
Qdrant collections.

Usage:
    python scripts/ingest_corpus.py [--verbose] [--dry-run] [--collection <name>]

Steps:
    1. Ensure Qdrant collections exist (creates them if not)
    2. Decompose Corpus/Diagnosis/*.md → C1: diagnostic_knowledge
    3. Decompose Corpus/Remediation/*.md → C4: remediation_actions
    4. Ingest knowledge/**/*.yaml → respective collections (C1-C6)
    5. Print ingestion summary

Environment:
    QDRANT_URL, HF_KEY must be set in .env or environment.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from airs.config import settings
from airs.retrieval.ingestion import (
    Collection,
    EmbeddingService,
    IngestionPipeline,
    create_ingestion_pipeline,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.DEBUG if "--verbose" in sys.argv else logging.INFO
    )
)
log = structlog.get_logger()


# ─── Qdrant Collection Setup ──────────────────────────────────────────────────

COLLECTION_CONFIG = {
    Collection.DIAGNOSTIC_KNOWLEDGE.value: {
        "description": "Diagnostic rules and heuristics (C1)",
        "indexed_fields": ["trigger_state", "signal_types", "failure_domain", "fault_categories"],
    },
    Collection.TOOL_SELECTION.value: {
        "description": "Tool profiles and selection guidance (C2)",
        "indexed_fields": ["signal_types", "tool_tier"],
    },
    Collection.QUERY_TEMPLATES.value: {
        "description": "Query syntax templates (C3)",
        "indexed_fields": ["applicable_tools"],
    },
    Collection.REMEDIATION_ACTIONS.value: {
        "description": "Remediation action procedures (C4)",
        "indexed_fields": ["trigger_state", "tool_tier", "fault_categories"],
    },
    Collection.FAILURE_SIGNATURES.value: {
        "description": "Failure pattern signatures (C5)",
        "indexed_fields": ["fault_categories", "failure_domain"],
    },
    Collection.OPERATIONAL_CONSTRAINTS.value: {
        "description": "Safety rules and operational constraints (C6)",
        "indexed_fields": ["applicable_tools", "trigger_state"],
    },
}


def ensure_collections(client: QdrantClient, recreate: bool = False) -> None:
    """Create all 6 Qdrant collections if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}

    for collection_name in Collection.all_names():
        if collection_name in existing:
            if recreate:
                log.info("Recreating collection", collection=collection_name)
                client.delete_collection(collection_name)
            else:
                log.info("Collection exists — skipping creation", collection=collection_name)
                continue

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        log.info(
            "Created collection",
            collection=collection_name,
            description=COLLECTION_CONFIG[collection_name]["description"],
        )


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest AIRS knowledge corpus into Qdrant"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log each document as it is ingested"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and decompose corpus but do not embed or upsert",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collections from scratch (deletes existing data)",
    )
    parser.add_argument(
        "--collection",
        choices=Collection.all_names(),
        help="Ingest only a specific collection (default: all)",
    )
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Skip Corpus/ Markdown ingestion (ingest only knowledge/ YAML)",
    )
    parser.add_argument(
        "--skip-yaml",
        action="store_true",
        help="Skip knowledge/ YAML ingestion",
    )
    args = parser.parse_args()

    # ── Validate configuration ────────────────────────────────────────────────
    if not settings.hf_key:
        log.error("HF_KEY not set — cannot embed documents. Set it in .env")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — documents will be parsed but not embedded or upserted")

    # ── Connect to Qdrant (skip in dry-run) ──────────────────────────────────
    client = None
    if not args.dry_run:
        log.info("Connecting to Qdrant", url=settings.qdrant_url)
        try:
            client = QdrantClient(url=settings.qdrant_url)
            client.get_collections()  # Test connection
        except Exception as e:
            log.error("Cannot connect to Qdrant", error=str(e))
            log.error("Start infrastructure first: make infra")
            sys.exit(1)
        ensure_collections(client, recreate=args.recreate)
    else:
        log.info("Dry-run mode — skipping Qdrant connection")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = create_ingestion_pipeline(
        qdrant_url=settings.qdrant_url,
        hf_key=settings.hf_key,
    )

    total_counts: dict[str, int] = {}

    # ── Step 1: Corpus/Diagnosis → C1 ────────────────────────────────────────
    if not args.skip_markdown and (
        args.collection is None
        or args.collection == Collection.DIAGNOSTIC_KNOWLEDGE.value
    ):
        diagnosis_dir = settings.corpus_diagnosis_dir
        if diagnosis_dir.exists():
            log.info("Ingesting Diagnosis corpus", directory=str(diagnosis_dir))
            if not args.dry_run:
                counts = pipeline.ingest_markdown_directory(
                    directory=diagnosis_dir,
                    collection=Collection.DIAGNOSTIC_KNOWLEDGE.value,
                    knowledge_type="HEURISTIC",
                    verbose=args.verbose,
                )
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            else:
                # Dry run: just count
                from airs.retrieval.ingestion import MarkdownDecomposer
                decomposer = MarkdownDecomposer()
                total = 0
                for md_file in sorted(diagnosis_dir.glob("*.md")):
                    docs = decomposer.decompose(
                        md_file,
                        Collection.DIAGNOSTIC_KNOWLEDGE.value,
                        "HEURISTIC",
                    )
                    total += len(docs)
                    log.info(
                        "  [DRY RUN] %s → %d atomic documents", md_file.name, len(docs)
                    )
                log.info("[DRY RUN] Total from Diagnosis: %d documents", total)
        else:
            log.warning("Diagnosis corpus not found", path=str(diagnosis_dir))

    # ── Step 2: Corpus/Remediation → C4 ──────────────────────────────────────
    if not args.skip_markdown and (
        args.collection is None
        or args.collection == Collection.REMEDIATION_ACTIONS.value
    ):
        remediation_dir = settings.corpus_remediation_dir
        if remediation_dir.exists():
            log.info("Ingesting Remediation corpus", directory=str(remediation_dir))
            if not args.dry_run:
                counts = pipeline.ingest_markdown_directory(
                    directory=remediation_dir,
                    collection=Collection.REMEDIATION_ACTIONS.value,
                    knowledge_type="PROCEDURAL",
                    verbose=args.verbose,
                )
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            else:
                from airs.retrieval.ingestion import MarkdownDecomposer
                decomposer = MarkdownDecomposer()
                total = 0
                for md_file in sorted(remediation_dir.glob("*.md")):
                    docs = decomposer.decompose(
                        md_file,
                        Collection.REMEDIATION_ACTIONS.value,
                        "PROCEDURAL",
                    )
                    total += len(docs)
                log.info("[DRY RUN] Total from Remediation: %d documents", total)
        else:
            log.warning("Remediation corpus not found", path=str(remediation_dir))

    # ── Step 3: knowledge/**/*.yaml → respective collections ─────────────────
    if not args.skip_yaml:
        knowledge_dir = settings.knowledge_dir
        if knowledge_dir.exists():
            log.info("Ingesting synthesized knowledge YAML", directory=str(knowledge_dir))
            if not args.dry_run:
                counts = pipeline.ingest_yaml_directory(
                    directory=knowledge_dir,
                    verbose=args.verbose,
                )
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            else:
                yaml_count = len(list(knowledge_dir.rglob("*.yaml")))
                log.info("[DRY RUN] Found %d YAML documents in knowledge/", yaml_count)
        else:
            log.info("knowledge/ directory not found — no YAML to ingest")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("INGESTION COMPLETE")
    if total_counts:
        for collection_name, count in sorted(total_counts.items()):
            log.info("  %-35s %d documents", collection_name, count)
        log.info("  TOTAL: %d documents", sum(total_counts.values()))
    elif args.dry_run:
        log.info("Dry run complete — no data written to Qdrant")
    else:
        log.info("No documents were ingested")


if __name__ == "__main__":
    main()
