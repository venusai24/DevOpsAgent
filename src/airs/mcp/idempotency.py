"""
Redis-backed Idempotency Manager — Module 1.6.

Prevents duplicate MCP tool invocations within the same investigation.
Every tool call is keyed by: investigation_id + hop_index + tool_name + query_hash.

This ensures that if a Temporal activity retries (due to network failure),
the MCP tool is NOT re-invoked — the cached response is returned instead.

Design:
    - Key:   er:idempotency:{investigation_id}:{hop_index}:{tool_name}:{query_hash}
    - Value: JSON-serialised MCPToolResponse (success, content, raw_digest)
    - TTL:   24 hours (investigation-scoped; expires after the investigation)
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import redis

log = logging.getLogger(__name__)

_KEY_TEMPLATE = "idem:{investigation_id}:{hop_index}:{tool_name}:{query_hash}"
_TTL_SECONDS = 86400  # 24 hours


class IdempotencyManager:
    """
    Redis-backed idempotency store for MCP tool invocations.

    Usage:
        manager = IdempotencyManager(redis_client)
        key = manager.make_key(inv_id, hop, tool, args)
        cached = manager.get(key)
        if cached:
            return cached  # Use cached response
        # ... invoke tool ...
        manager.set(key, response)
    """

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    def make_key(
        self,
        investigation_id: str,
        hop_index: int,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Generate a deterministic idempotency key for a tool invocation.

        The key encodes: investigation + hop + tool + query (content-hashed).
        This means retries with the same arguments always hit the same cache entry.
        """
        args_json = json.dumps(arguments, sort_keys=True)
        query_hash = hashlib.sha256(args_json.encode()).hexdigest()[:16]

        return _KEY_TEMPLATE.format(
            investigation_id=investigation_id,
            hop_index=hop_index,
            tool_name=tool_name,
            query_hash=query_hash,
        )

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """
        Look up a cached tool response.

        Returns:
            Deserialised response dict, or None if not found (or expired).
        """
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            log.warning("Idempotency cache read failed for key %s: %s", key, exc)
            return None

    def set(self, key: str, response: dict[str, Any]) -> None:
        """
        Cache a tool response.

        Args:
            key:      Idempotency key from make_key().
            response: Serialisable response dict.
        """
        try:
            self._redis.setex(key, self._ttl, json.dumps(response))
        except Exception as exc:
            log.warning("Idempotency cache write failed for key %s: %s", key, exc)

    def arguments_hash(self, arguments: dict[str, Any]) -> str:
        """Return the full SHA-256 hash of serialised arguments (for provenance)."""
        args_json = json.dumps(arguments, sort_keys=True)
        return hashlib.sha256(args_json.encode()).hexdigest()

    def is_cached(self, key: str) -> bool:
        """Check if a result is cached without fetching it."""
        try:
            return bool(self._redis.exists(key))
        except Exception:
            return False


def create_idempotency_manager(redis_url: str) -> IdempotencyManager:
    """Create an IdempotencyManager from a Redis URL."""
    client = redis.from_url(redis_url, decode_responses=False)
    return IdempotencyManager(client)
