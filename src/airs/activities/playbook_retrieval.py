"""
Playbook Retrieval Activity — Module 1.10.

Temporal activity: Query the ConANN retriever for diagnostic knowledge,
tool selection guidance, or operational constraints.

Returns formatted context string ready for injection into the next
LLM reasoning prompt.

Temporal contract:
  - Read-only: only reads from Qdrant — no state mutations
  - Returns formatted string (not EvidenceCandidate — playbook docs
    feed the LLM prompt, not the Investigation Graph)
  - Safe to retry without side effects
"""
from __future__ import annotations

import json
import logging

from temporalio import activity

from airs.models.intents import PlaybookQuery
from airs.retrieval.conann import ConANNResult, get_conann_retriever
from airs.retrieval.collections import build_payload_filter

log = logging.getLogger(__name__)

# Max token budget for playbook context in the prompt
_MAX_PLAYBOOK_TOKENS = 3000


@activity.defn(name="retrieve_playbook_context")
async def retrieve_playbook_context(
    query: PlaybookQuery,
) -> str:
    """
    Run a ConANN query against the specified knowledge collection.

    Args:
        query: PlaybookQuery with collection, query_text, filters, and k.

    Returns:
        Formatted playbook context string for LLM injection.
        Empty string if no conforming results found.
    """
    activity.logger.info(
        "Retrieving playbook: collection=%s, query=%s, k=%d",
        query.collection, query.query_text[:80], query.k,
    )

    retriever = get_conann_retriever()

    # Build payload filter from query.filters dict
    payload_filter = build_payload_filter(
        trigger_state=query.filters.get("trigger_state"),
        failure_domain=query.filters.get("failure_domain"),
        tool_tier=query.filters.get("tool_tier"),
        knowledge_type=query.filters.get("knowledge_type"),
    )

    try:
        results = retriever.retrieve(
            collection=query.collection,
            query_text=query.query_text,
            k=query.k,
            payload_filter=payload_filter,
        )
    except Exception as e:
        activity.logger.warning(
            "ConANN retrieval failed for collection=%s: %s — returning empty context",
            query.collection, e,
        )
        return ""

    if not results:
        activity.logger.info(
            "No conforming results from collection=%s for query=%s",
            query.collection, query.query_text[:80],
        )
        return ""

    formatted = _format_playbook_results(results, query.collection)

    activity.logger.info(
        "Playbook retrieved: %d results from %s (estimated %d tokens)",
        len(results), query.collection, len(formatted) // 4,
    )

    return formatted


def _format_playbook_results(
    results: list[ConANNResult],
    collection: str,
) -> str:
    """
    Format ConANN results as a structured text block for LLM injection.

    Budget: Caps total output at _MAX_PLAYBOOK_TOKENS (~12K chars).
    """
    lines = [
        f"## Playbook Context: {collection}",
        f"Retrieved {len(results)} conforming documents:\n",
    ]

    total_chars = 0
    char_budget = _MAX_PLAYBOOK_TOKENS * 4  # Approx chars per token

    for i, result in enumerate(results, 1):
        score_str = f"{result.score:.3f}"
        nonconf_str = f"{result.nonconformity:.3f}"
        title = result.payload.get("title", result.document_id)
        source = result.payload.get("source_file", "")

        header = (
            f"### [{i}] {title}\n"
            f"(source={source}, similarity={score_str}, nonconformity={nonconf_str})\n"
        )
        content = result.content or result.payload.get("content_preview", "")

        block = header + content + "\n\n"

        if total_chars + len(block) > char_budget:
            lines.append(f"[... {len(results) - i + 1} more results truncated for token budget ...]")
            break

        lines.append(block)
        total_chars += len(block)

    return "\n".join(lines)
