"""
Unit Tests — Entity Resolution Engine (Module 1.3).

Uses a real Redis instance (test database 15) for Union-Find operation tests.
Cleans up all test keys after each test via a dedicated Redis prefix flush.
"""
from __future__ import annotations

import uuid
from typing import Generator

import pytest
import redis

from airs.entity_resolution.engine import EntityResolutionEngine
from airs.models.evidence import EntityType

# Use Redis DB 15 for tests — isolated from dev DB 0
TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture(scope="module")
def redis_client() -> redis.Redis:
    """Module-scoped Redis client pointing at test DB."""
    client = redis.from_url(TEST_REDIS_URL, decode_responses=False)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available — skipping ER engine tests")
    yield client
    client.flushdb()  # Clean up test DB after module
    client.close()


@pytest.fixture
def er_engine(redis_client: redis.Redis) -> Generator[EntityResolutionEngine, None, None]:
    """Fresh ER engine per test; flushes DB before each test."""
    redis_client.flushdb()
    yield EntityResolutionEngine(redis_client, ttl_seconds=300)


class TestEntityResolutionEngine:

    def test_fresh_entity_gets_guid(self, er_engine: EntityResolutionEngine) -> None:
        guid = er_engine.resolve_entity(
            EntityType.SERVICE,
            ["frontend"],
        )
        assert isinstance(guid, uuid.UUID)

    def test_same_identifiers_same_guid(self, er_engine: EntityResolutionEngine) -> None:
        guid1 = er_engine.resolve_entity(EntityType.SERVICE, ["frontend"])
        guid2 = er_engine.resolve_entity(EntityType.SERVICE, ["frontend"])
        assert guid1 == guid2

    def test_different_types_different_guids(
        self, er_engine: EntityResolutionEngine
    ) -> None:
        service_guid = er_engine.resolve_entity(EntityType.SERVICE, ["frontend"])
        pod_guid = er_engine.resolve_entity(EntityType.POD, ["frontend"])
        assert service_guid != pod_guid

    def test_alias_stitching_resolves_to_same_guid(
        self, er_engine: EntityResolutionEngine
    ) -> None:
        """
        Simulate discovering that container-id 'abc123' belongs to
        pod 'frontend-pod-xyz'. Both should resolve to the same canonical GUID.
        """
        # First encounter: only the container ID is known
        guid1 = er_engine.resolve_entity(EntityType.POD, ["abc123"])
        # Later: we learn pod name maps to the same entity
        guid2 = er_engine.resolve_entity(EntityType.POD, ["abc123", "frontend-pod-xyz"])
        assert guid1 == guid2

    def test_union_merges_two_previously_separate_entities(
        self, er_engine: EntityResolutionEngine
    ) -> None:
        """
        Two separate canonical entities discovered to be the same should be merged.
        """
        guid_a = er_engine.resolve_entity(EntityType.SERVICE, ["svc-a"])
        guid_b = er_engine.resolve_entity(EntityType.SERVICE, ["svc-b"])
        assert guid_a != guid_b  # Initially separate

        # Now discover they are the same service (seen together in one telemetry record)
        guid_merged = er_engine.resolve_entity(EntityType.SERVICE, ["svc-a", "svc-b"])
        # Both original GUIDs should now resolve to the same root
        root_a = er_engine._find_root(guid_a)
        root_b = er_engine._find_root(guid_b)
        assert root_a == root_b
        assert guid_merged in (root_a, root_b)

    def test_deterministic_guid_for_same_unsorted_identifiers(
        self, er_engine: EntityResolutionEngine
    ) -> None:
        """Identifier order should not affect the produced GUID."""
        guid1 = er_engine.resolve_entity(EntityType.SERVICE, ["alpha", "beta", "gamma"])
        er_engine._redis.flushdb()  # Clear all state
        guid2 = er_engine.resolve_entity(EntityType.SERVICE, ["gamma", "alpha", "beta"])
        assert guid1 == guid2

    def test_get_canonical_unknown_identifier_returns_none(
        self, er_engine: EntityResolutionEngine
    ) -> None:
        result = er_engine.get_canonical(EntityType.SERVICE, "never-seen")
        assert result is None

    def test_get_canonical_known_identifier(
        self, er_engine: EntityResolutionEngine
    ) -> None:
        created = er_engine.resolve_entity(EntityType.DATABASE, ["postgres-primary"])
        looked_up = er_engine.get_canonical(EntityType.DATABASE, "postgres-primary")
        assert looked_up is not None
        assert er_engine._find_root(created) == er_engine._find_root(looked_up)

    def test_empty_identifiers_raises(self, er_engine: EntityResolutionEngine) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            er_engine.resolve_entity(EntityType.SERVICE, [])
