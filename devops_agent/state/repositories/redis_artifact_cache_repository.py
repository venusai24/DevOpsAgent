"""Concrete ArtifactCacheRepository backed by Redis."""

from __future__ import annotations

import logging
from typing import Any

from ..adapters.redis_adapter import RedisAdapter
from ..interfaces.artifact_cache_repository import (
    ARTIFACT_CACHE_TTL_SECONDS,
    ArtifactCacheRepository,
)

logger = logging.getLogger(__name__)

_IN_PROGRESS_SUFFIX = ":in_progress"
_IN_PROGRESS_TTL = 300  # 5 minutes


class RedisArtifactCacheRepository(ArtifactCacheRepository):
    """Redis-backed Stage 0 Artifact Cache.

    Cache keys are version-keyed per §3.5.1 — staleness is structurally
    unreachable, not actively invalidated.

    The in_progress sentinel (§9.3) uses SET NX for atomic single-writer
    election among concurrent workers.
    """

    def __init__(self, adapter: RedisAdapter):
        self._redis = adapter

    async def get(self, key: str) -> dict[str, Any] | None:
        return await self._redis.get_json(key)

    async def set(
        self,
        key: str,
        payload: dict[str, Any],
        ttl_seconds: int = ARTIFACT_CACHE_TTL_SECONDS,
    ) -> None:
        await self._redis.set_json(key, payload, ttl_seconds=ttl_seconds)
        logger.debug("ArtifactCache SET %s (TTL=%ds)", key, ttl_seconds)

    async def set_in_progress(self, key: str, ttl_seconds: int = _IN_PROGRESS_TTL) -> bool:
        sentinel_key = key + _IN_PROGRESS_SUFFIX
        acquired = await self._redis.set_nx(sentinel_key, "1", ttl_seconds)
        logger.debug(
            "ArtifactCache in_progress sentinel for %s: acquired=%s", key, acquired
        )
        return acquired

    async def clear_in_progress(self, key: str) -> None:
        await self._redis.delete(key + _IN_PROGRESS_SUFFIX)

    async def is_in_progress(self, key: str) -> bool:
        return await self._redis.exists(key + _IN_PROGRESS_SUFFIX)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def ttl(self, key: str) -> int | None:
        return await self._redis.ttl(key)
