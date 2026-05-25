"""
agent/kb/store.py

Qdrant-backed retrieval engine for the AIRS Knowledge Base.

Architecture
------------
The store uses a two-stage pipeline layered on Qdrant (local Docker or
Qdrant Cloud):

  Stage 1 — Vector Search (Qdrant, sub-millisecond):
    The combined alert + telemetry text is embedded with the HuggingFace
    ``sentence-transformers/all-MiniLM-L6-v2`` model (384 dimensions, cosine
    distance).  Qdrant returns the top-K (default: 5) nearest neighbours by
    cosine similarity.  These are the only candidates evaluated further.

  Stage 2 — LLM Reranker (Groq, on top-K only):
    For each candidate whose Qdrant cosine score is below the fast-path
    threshold (0.85), the Groq LLM is asked to score semantic relevance
    0.0–1.0.  The final score is:

        final_score = max(qdrant_score, llm_score) * entry.confidence_score

    Candidates above the fast-path threshold (qdrant_score >= 0.85) skip the
    LLM call entirely and use the Qdrant score directly, weighted by
    confidence_score.

Score thresholds (from config.py):
  >= KB_EXACT_BYPASS_THRESHOLD (0.95) → bypass_llm=True  (full automation)
  >= KB_RAG_THRESHOLD          (0.70) → RAG context injected into plan_node
  <  KB_RAG_THRESHOLD                 → read-only diagnostic mode

Collection schema
-----------------
Each Qdrant point stores:
  id:      deterministic UUID derived from entry.entry_id (str → uuid5)
  vector:  384-dim float32 cosine vector of (error_pattern + root_cause_narrative)
  payload: full KBEntry JSON as a flat dict (entry.model_dump())

The payload contains the canonical ``entry_id`` string so we can look up
the original entry for confidence updates.

Public API
----------
    from agent.kb.store import kb_lookup, kb_insert, kb_update_confidence

    result: KBRetrievalResult = await kb_lookup(title, description, evidence)
    await kb_insert(entry)
    await kb_update_confidence(entry_id, success=True)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from agent.kb.embedder import embed_text, embed_for_query, VECTOR_DIM
from agent.state import KBEntry, KBRetrievalResult

logger = logging.getLogger(__name__)

# Qdrant cosine score above which we skip the LLM reranker entirely.
# 0.85 is a strong semantic match for all-MiniLM-L6-v2 cosine distance.
_FAST_PATH_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Lazy config / client helpers
# ---------------------------------------------------------------------------


def _settings():
    from config import settings
    return settings


def _get_qdrant_client() -> QdrantClient:
    """Open a QdrantClient connected to the configured QDRANT_URL."""
    cfg = _settings()
    return QdrantClient(url=cfg.QDRANT_URL)


def _entry_id_to_point_id(entry_id: str) -> str:
    """
    Convert an arbitrary entry_id string (e.g. 'kb-001') to a deterministic
    UUID string that Qdrant accepts as a point ID.

    Uses uuid5 with the DNS namespace so the mapping is stable across restarts.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_id))


# ---------------------------------------------------------------------------
# Collection bootstrap
# ---------------------------------------------------------------------------


def _ensure_collection(client: QdrantClient) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Uses cosine distance to match the normalised all-MiniLM-L6-v2 vectors.
    Idempotent: safe to call on every insert.
    """
    cfg = _settings()
    collection_name = cfg.QDRANT_COLLECTION

    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "[kb_store] Created Qdrant collection '%s' (dim=%d, cosine)",
            collection_name, VECTOR_DIM,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def kb_lookup(
    alert_title: str,
    alert_description: str = "",
    extracted_evidence: str = "",
) -> KBRetrievalResult:
    """
    Find the best-matching KB entry for the given alert and telemetry.

    Stage 1: Embeds the combined query text and performs a cosine vector
    search in Qdrant, retrieving up to KB_TOP_K candidates.

    Stage 2: For candidates below the fast-path threshold, calls the Groq
    LLM to rerank.  Score-fuses the result with entry.confidence_score.

    Args:
        alert_title:        Alert title from state["raw_alert"]["title"].
        alert_description:  Alert description from state["raw_alert"]["description"].
        extracted_evidence: Verbatim evidence from extract_node (may be empty
                            on first KB lookup before investigation runs).

    Returns:
        KBRetrievalResult with the best-matching entry and its routing flags.
    """
    cfg = _settings()
    combined_text = f"{alert_title} {alert_description} {extracted_evidence}".strip()
    alert_text = f"{alert_title} {alert_description}"

    # Stage 1: embed query + Qdrant ANN search
    query_vector = await embed_for_query(combined_text)
    client = _get_qdrant_client()

    try:
        search_result = client.query_points(
            collection_name=cfg.QDRANT_COLLECTION,
            query=query_vector,
            limit=cfg.KB_TOP_K,
            with_payload=True,
        )
        hits = search_result.points
    except Exception as exc:
        logger.error("[kb_store] Qdrant search failed: %s", exc)
        return KBRetrievalResult(
            entry=None,
            retrieval_score=0.0,
            match_type="none",
            bypass_llm=False,
        )

    if not hits:
        logger.info("[kb_store] No Qdrant candidates returned for query.")
        return KBRetrievalResult(
            entry=None,
            retrieval_score=0.0,
            match_type="none",
            bypass_llm=False,
        )

    logger.info("[kb_store] Qdrant returned %d candidate(s)", len(hits))

    # Stage 2: rerank top-K with LLM (only for candidates below fast-path)
    best_score: float = 0.0
    best_entry: KBEntry | None = None
    best_match_type: str = "none"

    for hit in hits:
        qdrant_score: float = float(hit.score)  # cosine similarity 0.0–1.0

        try:
            entry = KBEntry.model_validate(hit.payload)
        except Exception as exc:
            logger.warning("[kb_store] Skipping malformed payload: %s", exc)
            continue

        if qdrant_score >= _FAST_PATH_THRESHOLD:
            # Strong semantic hit — skip LLM call
            final_score = qdrant_score * entry.confidence_score
            match_type = "semantic"
            logger.debug(
                "[kb_store] fast-path entry=%s qdrant=%.3f conf=%.3f final=%.3f",
                entry.entry_id, qdrant_score, entry.confidence_score, final_score,
            )
        else:
            # Moderate similarity — ask LLM to rerank
            llm_score = await _llm_judge_similarity(alert_text, entry)
            final_score = max(qdrant_score, llm_score) * entry.confidence_score
            match_type = "semantic"
            logger.debug(
                "[kb_store] llm-rerank entry=%s qdrant=%.3f llm=%.3f conf=%.3f final=%.3f",
                entry.entry_id, qdrant_score, llm_score,
                entry.confidence_score, final_score,
            )

        if final_score > best_score:
            best_score = final_score
            best_entry = entry
            best_match_type = match_type

    # Clamp to [0.0, 1.0]
    best_score = min(1.0, max(0.0, best_score))

    if best_score < cfg.KB_RAG_THRESHOLD or best_entry is None:
        logger.info(
            "[kb_store] No KB match above RAG threshold (best=%.3f, threshold=%.2f)",
            best_score, cfg.KB_RAG_THRESHOLD,
        )
        return KBRetrievalResult(
            entry=None,
            retrieval_score=best_score,
            match_type="none",
            bypass_llm=False,
        )

    bypass = best_score >= cfg.KB_EXACT_BYPASS_THRESHOLD
    logger.info(
        "[kb_store] Best match: entry=%s score=%.3f match_type=%s bypass_llm=%s",
        best_entry.entry_id, best_score, best_match_type, bypass,
    )
    return KBRetrievalResult(
        entry=best_entry,
        retrieval_score=best_score,
        match_type=best_match_type,  # type: ignore[arg-type]
        bypass_llm=bypass,
    )


async def kb_insert(entry: KBEntry) -> None:
    """
    Persist a new KBEntry to the Qdrant store.

    Embeds the entry's ``error_pattern + root_cause_narrative`` into a
    384-dim vector and upserts a PointStruct into the collection.  The full
    KBEntry JSON is stored in the point payload.

    Idempotent: if a point with the same derived UUID already exists, Qdrant
    upserts (overwrites) it rather than raising an error.

    Args:
        entry: The KBEntry Pydantic model to persist.
    """
    cfg = _settings()
    client = _get_qdrant_client()
    _ensure_collection(client)

    # Embed the diagnostic signal text
    embed_text_input = f"{entry.error_pattern} {entry.root_cause_narrative}"
    try:
        vector = await embed_text(embed_text_input)
    except Exception as exc:
        logger.error(
            "[kb_store] Embedding failed for entry %s: %s — skipping insert",
            entry.entry_id, exc,
        )
        return

    point_id = _entry_id_to_point_id(entry.entry_id)

    try:
        client.upsert(
            cfg.QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=entry.model_dump(),  # Full KBEntry JSON in payload
                )
            ],
        )
        logger.info(
            "[kb_store] Upserted entry: %s (point_id=%s)", entry.entry_id, point_id
        )
    except Exception as exc:
        logger.error(
            "[kb_store] Failed to upsert entry %s: %s", entry.entry_id, exc
        )


async def kb_update_confidence(entry_id: str, success: bool) -> None:
    """
    Update the confidence_score of an existing KB entry based on execution
    outcome.

    Increments by 0.01 on success, decrements by 0.02 on failure, bounded
    to [0.0, 1.0].  Fetches the current payload from Qdrant, computes the new
    score in Python, then writes it back via ``set_payload``.

    Args:
        entry_id: The KB entry's entry_id string (e.g. 'kb-001').
        success:  True if the remediation was verified successful.
    """
    cfg = _settings()
    client = _get_qdrant_client()
    point_id = _entry_id_to_point_id(entry_id)

    try:
        # Fetch the current point payload
        results, _ = client.scroll(
            collection_name=cfg.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="entry_id",
                        match=MatchValue(value=entry_id),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=1,
        )

        if not results:
            logger.warning(
                "[kb_store] entry_id=%s not found in Qdrant for confidence update",
                entry_id,
            )
            return

        payload = results[0].payload
        current_score: float = float(payload.get("confidence_score", 0.80))

        if success:
            new_score = min(1.0, round(current_score + 0.01, 4))
        else:
            new_score = max(0.0, round(current_score - 0.02, 4))

        # Build updated payload — update both the top-level field and the
        # nested entry so _load or model_validate always sees fresh data.
        client.set_payload(
            collection_name=cfg.QDRANT_COLLECTION,
            payload={"confidence_score": new_score},
            points=[point_id],
        )

        logger.info(
            "[kb_store] Updated confidence for entry=%s: %.4f → %.4f (success=%s)",
            entry_id, current_score, new_score, success,
        )
    except Exception as exc:
        logger.error(
            "[kb_store] Failed to update confidence for %s: %s", entry_id, exc
        )


def kb_count() -> int:
    """Return the total number of points in the KB collection. Useful for health checks."""
    cfg = _settings()
    try:
        client = _get_qdrant_client()
        result = client.count(collection_name=cfg.QDRANT_COLLECTION, exact=True)
        return result.count
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Private: LLM reranker
# ---------------------------------------------------------------------------


async def _llm_judge_similarity(alert_text: str, entry: KBEntry) -> float:
    """
    Use the Groq LLM to score how closely the alert text matches the KB entry.

    Called only for candidates below ``_FAST_PATH_THRESHOLD`` (0.85).
    Temperature=0 and a tight single-number prompt keep latency and cost low.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 on any API or parse failure so the
        Qdrant cosine score is used as the fallback.
    """
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    cfg = _settings()
    llm = ChatGroq(model=cfg.GROQ_MODEL, temperature=0, max_retries=2)

    prompt = (
        "Rate the semantic similarity between the following incident alert and "
        "knowledge base entry on a scale from 0.0 (completely unrelated) to "
        "1.0 (exact match).\n\n"
        f"**Incident Alert**: {alert_text[:400]}\n\n"
        f"**KB Entry Pattern**: {entry.error_pattern}\n"
        f"**KB Root Cause Summary**: {entry.root_cause_narrative[:250]}\n\n"
        "Respond with ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation. No other text."
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        score_text = response.content.strip().split()[0]
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning(
            "[kb_store] LLM reranker failed for entry %s: %s", entry.entry_id, exc
        )
        return 0.0
