"""Abstract base for all state serializers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T")


class BaseSerializer[T](ABC):
    """Contract for all state serializers.

    - ``serialise`` converts a state model to a JSON-safe dict.
    - ``deserialise`` converts a JSON-safe dict back to a state model.

    Both operations must be pure functions (no I/O, no side-effects).
    """

    @abstractmethod
    def serialise(self, state: T) -> dict[str, Any]:
        """Convert ``state`` to a JSON-safe dict for persistence."""

    @abstractmethod
    def deserialise(self, data: dict[str, Any]) -> T:
        """Reconstruct a state model from a persisted dict."""

    def serialise_or_none(self, state: T | None) -> dict[str, Any] | None:
        if state is None:
            return None
        return self.serialise(state)
