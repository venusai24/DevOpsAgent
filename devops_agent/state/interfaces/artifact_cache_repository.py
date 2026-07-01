"""Abstract repository for the Stage 0 Artifact Cache (Redis/DynamoDB).

Cache key format: stage0:{cluster_id}:{time_window_hash}:{manifest_version}
TTL: 6 hours (Phase 3 §3.3.4).
Version-keyed to make stale entries unreachable on manifest change (§3.5.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

ARTIFACT_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


class ArtifactCacheRepository(ABC):
    """Controls read/write access to the Stage 0 Artifact Cache."""

    @staticmethod
    def make_key(
        cluster_id: str,
        time_window_hash: str,
        manifest_version: str,
    ) -> str:
        """Build the canonical version-keyed cache key."""
        return f"stage0:{cluster_id}:{time_window_hash}:{manifest_version}"

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached artifact bundle, or None on miss."""

    @abstractmethod
    async def set(
        self,
        key: str,
        payload: dict[str, Any],
        ttl_seconds: int = ARTIFACT_CACHE_TTL_SECONDS,
    ) -> None:
        """Write an artifact bundle with the given TTL."""

    @abstractmethod
    async def set_in_progress(self, key: str, ttl_seconds: int = 300) -> bool:
        """Attempt to atomically set an in_progress sentinel.

        Returns True when the sentinel was set (this caller owns the compute);
        returns False when another caller already holds it (§9.3 backoff logic).
        """

    @abstractmethod
    async def clear_in_progress(self, key: str) -> None:
        """Remove the in_progress sentinel."""

    @abstractmethod
    async def is_in_progress(self, key: str) -> bool:
        """Return True when an in_progress sentinel is active."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Explicitly delete a cache entry (e.g., on staleness invalidation)."""

    @abstractmethod
    async def ttl(self, key: str) -> int | None:
        """Return remaining TTL in seconds, or None when key does not exist."""
