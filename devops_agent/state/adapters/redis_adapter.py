"""Redis connection pool and pipeline management adapter.

Used exclusively for the Stage 0 Artifact Cache (ArtifactCacheRepository).
All I/O is async via aioredis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_ENCODING = "utf-8"


class RedisAdapter:
    """Wraps aioredis to provide typed get/set/delete with JSON serialisation.

    Usage::

        adapter = RedisAdapter(url="redis://localhost:6379/0")
        await adapter.connect()

        await adapter.set_json("key", {"field": "value"}, ttl_seconds=3600)
        data = await adapter.get_json("key")

        await adapter.disconnect()
    """

    def __init__(self, url: str, decode_responses: bool = True, max_connections: int = 20):
        self._url = url
        self._decode_responses = decode_responses
        self._max_connections = max_connections
        self._redis: Redis | None = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            self._url,
            encoding=_ENCODING,
            decode_responses=self._decode_responses,
            max_connections=self._max_connections,
        )
        logger.info("RedisAdapter connected to %s", self._url)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("RedisAdapter disconnected")

    @property
    def redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("RedisAdapter.connect() has not been called")
        return self._redis

    # ── JSON helpers ─────────────────────────────────────────────────────────

    async def get_json(self, key: str) -> dict[str, Any] | None:
        raw = await self.redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(
        self, key: str, value: dict[str, Any], ttl_seconds: int | None = None
    ) -> None:
        raw = json.dumps(value, separators=(",", ":"))
        if ttl_seconds is not None:
            await self.redis.set(key, raw, ex=ttl_seconds)
        else:
            await self.redis.set(key, raw)

    async def delete(self, key: str) -> None:
        await self.redis.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self.redis.exists(key))

    async def ttl(self, key: str) -> int | None:
        t = await self.redis.ttl(key)
        return None if t < 0 else t

    # ── Atomic set-if-not-exists (in_progress sentinel) ──────────────────────

    async def set_nx(
        self, key: str, value: str, ttl_seconds: int
    ) -> bool:
        """SET NX with expiry.  Returns True if the key was set."""
        result = await self.redis.set(key, value, ex=ttl_seconds, nx=True)
        return result is not None
