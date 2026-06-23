"""
Qdrant Collection Manager — Module 1.8.

Handles creation, validation, and schema management of all 6 Qdrant
collections for the AIRS knowledge corpus.

Collection schema:
  C1: diagnostic_knowledge     — HEURISTIC rules, indexed by trigger_state, signal_types, failure_domain
  C2: tool_selection           — DECLARATIVE tool profiles, indexed by signal_types, tool_tier
  C3: query_templates          — PROCEDURAL query templates, indexed by applicable_tools
  C4: remediation_actions      — PROCEDURAL remediation steps, indexed by trigger_state, tool_tier
  C5: failure_signatures       — HEURISTIC failure pattern fingerprints, indexed by fault_categories
  C6: operational_constraints  — NORMATIVE safety rules, indexed by applicable_tools, trigger_state
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

log = logging.getLogger(__name__)

EMBEDDING_DIM = 768
DISTANCE = Distance.COSINE

# All 6 collection names — single source of truth
ALL_COLLECTIONS = [
    "diagnostic_knowledge",
    "tool_selection",
    "query_templates",
    "remediation_actions",
    "failure_signatures",
    "operational_constraints",
]

# Per-collection indexed payload fields (for pre-filter efficiency)
_INDEXED_FIELDS: dict[str, list[tuple[str, PayloadSchemaType]]] = {
    "diagnostic_knowledge": [
        ("trigger_state", PayloadSchemaType.KEYWORD),
        ("failure_domain", PayloadSchemaType.KEYWORD),
        ("knowledge_type", PayloadSchemaType.KEYWORD),
    ],
    "tool_selection": [
        ("tool_tier", PayloadSchemaType.INTEGER),
        ("knowledge_type", PayloadSchemaType.KEYWORD),
    ],
    "query_templates": [
        ("knowledge_type", PayloadSchemaType.KEYWORD),
    ],
    "remediation_actions": [
        ("trigger_state", PayloadSchemaType.KEYWORD),
        ("tool_tier", PayloadSchemaType.INTEGER),
        ("knowledge_type", PayloadSchemaType.KEYWORD),
    ],
    "failure_signatures": [
        ("failure_domain", PayloadSchemaType.KEYWORD),
        ("knowledge_type", PayloadSchemaType.KEYWORD),
    ],
    "operational_constraints": [
        ("trigger_state", PayloadSchemaType.KEYWORD),
        ("knowledge_type", PayloadSchemaType.KEYWORD),
    ],
}


class CollectionManager:
    """
    Manages the lifecycle of all 6 Qdrant collections.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def ensure_all(self, recreate: bool = False) -> None:
        """Create all 6 collections if they don't exist (or recreate if requested)."""
        existing = {c.name for c in self._client.get_collections().collections}

        for name in ALL_COLLECTIONS:
            if name in existing and not recreate:
                log.debug("Collection '%s' exists — skipping", name)
                continue

            if name in existing and recreate:
                log.info("Recreating collection '%s'", name)
                self._client.delete_collection(name)

            self._create_collection(name)

    def _create_collection(self, name: str) -> None:
        """Create a single collection with its vector config."""
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=DISTANCE,
            ),
        )

        # Create payload indexes for efficient pre-filtering
        for field_name, field_type in _INDEXED_FIELDS.get(name, []):
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                log.warning("Failed to create index %s.%s: %s", name, field_name, e)

        log.info("Created collection '%s' (dim=%d, distance=COSINE)", name, EMBEDDING_DIM)

    def collection_stats(self) -> dict[str, dict]:
        """Return point counts for all collections."""
        stats = {}
        for name in ALL_COLLECTIONS:
            try:
                info = self._client.get_collection(name)
                stats[name] = {
                    "vectors_count": info.vectors_count or 0,
                    "points_count": info.points_count or 0,
                    "status": info.status,
                }
            except Exception:
                stats[name] = {"vectors_count": 0, "points_count": 0, "status": "missing"}
        return stats

    def validate_populated(self) -> dict[str, bool]:
        """Return {collection: is_populated} for all 6 collections."""
        stats = self.collection_stats()
        return {name: stats.get(name, {}).get("points_count", 0) > 0 for name in ALL_COLLECTIONS}


def build_payload_filter(
    trigger_state: Optional[str] = None,
    failure_domain: Optional[str] = None,
    tool_tier: Optional[int] = None,
    signal_types: Optional[list[str]] = None,
    knowledge_type: Optional[str] = None,
) -> Optional[Filter]:
    """
    Build a Qdrant payload pre-filter from structured criteria.

    Only non-None criteria are included. Returns None if all criteria are None
    (meaning no pre-filter — full collection search).

    Args:
        trigger_state:   e.g., 'Continue', 'Escalate'
        failure_domain:  e.g., 'data_tier', 'network'
        tool_tier:       1, 2, or 3
        signal_types:    e.g., ['METRICS', 'LOGS']
        knowledge_type:  e.g., 'HEURISTIC', 'PROCEDURAL'

    Returns:
        Qdrant Filter or None.
    """
    conditions = []

    if trigger_state:
        conditions.append(
            FieldCondition(key="trigger_state", match=MatchValue(value=trigger_state))
        )
    if failure_domain:
        conditions.append(
            FieldCondition(key="failure_domain", match=MatchValue(value=failure_domain))
        )
    if tool_tier is not None:
        conditions.append(
            FieldCondition(key="tool_tier", match=MatchValue(value=tool_tier))
        )
    if knowledge_type:
        conditions.append(
            FieldCondition(key="knowledge_type", match=MatchValue(value=knowledge_type))
        )

    if not conditions:
        return None

    from qdrant_client.models import Filter as QFilter
    return QFilter(must=conditions)
