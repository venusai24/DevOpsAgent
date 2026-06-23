"""
Entity Resolution Engine — Module 1.3.

Resolves raw entity identifiers (container IDs, pod names, service names)
to stable canonical GUIDs using Union-Find backed by Redis.

This prevents the Investigation Graph from creating duplicate nodes for the
same logical entity observed through different telemetry signals (e.g., a
container ID in logs vs. a pod name in metrics both referring to the same pod).
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Optional

import redis

from airs.models.evidence import EntityType

log = logging.getLogger(__name__)

# Redis key patterns
_ALIAS_KEY = "er:alias:{entity_type}:{identifier}"
_CANONICAL_KEY = "er:canonical:{canonical_guid}"
_PARENT_KEY = "er:parent:{canonical_guid}"


class EntityResolutionEngine:
    """
    Union-Find entity resolution engine backed by Redis.

    Produces a stable canonical GUID for any combination of entity type +
    raw identifiers. Merges entities when new aliases are discovered.

    Thread-safe: all state lives in Redis. Multiple workers can share
    the same engine without coordination overhead.
    """

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 86400) -> None:
        """
        Args:
            redis_client: Connected Redis client.
            ttl_seconds:  TTL for all ER keys (default: 24 hours).
                          Expired aliases are re-created on next encounter.
        """
        self._redis = redis_client
        self._ttl = ttl_seconds

    # ─── Public API ───────────────────────────────────────────────────────────

    def resolve_entity(
        self,
        entity_type: EntityType,
        raw_identifiers: list[str],
    ) -> uuid.UUID:
        """
        Resolve a set of raw identifiers to a canonical GUID.

        If all identifiers are new, creates a fresh canonical entity.
        If any identifier already maps to a canonical entity, uses that.
        If identifiers map to multiple canonical entities, merges them
        (Union-Find union operation).

        Args:
            entity_type:      Classification of the entity.
            raw_identifiers:  List of raw identifier strings (e.g., container
                              ID, pod name, service name).

        Returns:
            Stable UUID for the canonical entity.
        """
        if not raw_identifiers:
            raise ValueError("raw_identifiers must not be empty")

        # Step 1: Look up all existing canonical GUIDs for these identifiers
        existing_guids: list[uuid.UUID] = []
        for identifier in raw_identifiers:
            alias_key = _ALIAS_KEY.format(
                entity_type=entity_type.value, identifier=identifier
            )
            existing = self._redis.get(alias_key)
            if existing:
                root_guid = self._find_root(uuid.UUID(existing.decode()))
                existing_guids.append(root_guid)

        # Step 2: Determine canonical GUID
        if not existing_guids:
            # All identifiers are new — allocate a fresh GUID
            canonical_guid = self._allocate_guid(entity_type, raw_identifiers)
        else:
            # Use the first found GUID as the canonical root
            canonical_guid = existing_guids[0]
            # Merge any additional roots (union operation)
            for other_guid in existing_guids[1:]:
                if other_guid != canonical_guid:
                    self._union(canonical_guid, other_guid)
                    canonical_guid = self._find_root(canonical_guid)

        # Step 3: Record all identifiers as aliases of the canonical GUID
        pipe = self._redis.pipeline()
        for identifier in raw_identifiers:
            alias_key = _ALIAS_KEY.format(
                entity_type=entity_type.value, identifier=identifier
            )
            pipe.setex(alias_key, self._ttl, str(canonical_guid))
        pipe.execute()

        return canonical_guid

    def get_canonical(
        self,
        entity_type: EntityType,
        identifier: str,
    ) -> Optional[uuid.UUID]:
        """
        Look up the canonical GUID for a single identifier without creating one.

        Returns None if the identifier is not yet known to the engine.
        """
        alias_key = _ALIAS_KEY.format(
            entity_type=entity_type.value, identifier=identifier
        )
        existing = self._redis.get(alias_key)
        if existing is None:
            return None
        root = self._find_root(uuid.UUID(existing.decode()))
        return root

    def get_observed_identifiers(self, canonical_guid: uuid.UUID) -> list[str]:
        """
        Retrieve all observed identifiers for a canonical entity.

        Note: This is an O(n) scan — use only for debugging/reporting.
        """
        canonical_key = _CANONICAL_KEY.format(canonical_guid=str(canonical_guid))
        root = self._find_root(canonical_guid)
        canonical_key = _CANONICAL_KEY.format(canonical_guid=str(root))
        aliases = self._redis.smembers(canonical_key)
        return [a.decode() for a in aliases]

    # ─── Union-Find internals ─────────────────────────────────────────────────

    def _find_root(self, guid: uuid.UUID) -> uuid.UUID:
        """Path-compressed find operation."""
        parent_key = _PARENT_KEY.format(canonical_guid=str(guid))
        parent_raw = self._redis.get(parent_key)

        if parent_raw is None or parent_raw.decode() == str(guid):
            # guid IS the root
            return guid

        # Recurse to find root
        parent = uuid.UUID(parent_raw.decode())
        root = self._find_root(parent)

        # Path compression — point directly to root
        if root != parent:
            self._redis.setex(parent_key, self._ttl, str(root))

        return root

    def _union(self, guid_a: uuid.UUID, guid_b: uuid.UUID) -> None:
        """Union two canonical entities (guid_b's tree becomes child of guid_a)."""
        root_a = self._find_root(guid_a)
        root_b = self._find_root(guid_b)

        if root_a == root_b:
            return  # Already same entity

        # Point root_b to root_a
        parent_b_key = _PARENT_KEY.format(canonical_guid=str(root_b))
        self._redis.setex(parent_b_key, self._ttl, str(root_a))

        # Merge alias sets: move root_b's aliases into root_a's set
        canonical_b_key = _CANONICAL_KEY.format(canonical_guid=str(root_b))
        canonical_a_key = _CANONICAL_KEY.format(canonical_guid=str(root_a))

        aliases_b = self._redis.smembers(canonical_b_key)
        if aliases_b:
            pipe = self._redis.pipeline()
            for alias in aliases_b:
                pipe.sadd(canonical_a_key, alias)
            pipe.delete(canonical_b_key)
            pipe.expire(canonical_a_key, self._ttl)
            pipe.execute()

        log.debug("ER: merged %s → %s", root_b, root_a)

    def _allocate_guid(
        self, entity_type: EntityType, raw_identifiers: list[str]
    ) -> uuid.UUID:
        """
        Allocate a new deterministic canonical GUID.

        Uses uuid5 with a namespace derived from entity_type + sorted identifiers.
        This guarantees that the same set of identifiers always produces the same
        GUID, even across process restarts or in a fresh Redis instance.
        """
        namespace = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"airs.entity.{entity_type.value}",
        )
        # Sort identifiers for determinism regardless of input order
        sorted_key = "|".join(sorted(raw_identifiers))
        canonical_guid = uuid.uuid5(namespace, sorted_key)

        # Register in the canonical set
        canonical_key = _CANONICAL_KEY.format(canonical_guid=str(canonical_guid))
        parent_key = _PARENT_KEY.format(canonical_guid=str(canonical_guid))

        pipe = self._redis.pipeline()
        for ident in raw_identifiers:
            pipe.sadd(canonical_key, ident)
        # Self-referential parent means "I am the root"
        pipe.setex(parent_key, self._ttl, str(canonical_guid))
        pipe.expire(canonical_key, self._ttl)
        pipe.execute()

        log.debug(
            "ER: allocated %s for %s identifiers: %s",
            canonical_guid,
            entity_type.value,
            raw_identifiers,
        )
        return canonical_guid


# ─── Factory helper ───────────────────────────────────────────────────────────

def create_er_engine(redis_url: str, ttl_seconds: int = 86400) -> EntityResolutionEngine:
    """
    Create an EntityResolutionEngine from a Redis URL.

    Usage:
        engine = create_er_engine(settings.redis_url)
    """
    client = redis.from_url(redis_url, decode_responses=False)
    return EntityResolutionEngine(client, ttl_seconds=ttl_seconds)
