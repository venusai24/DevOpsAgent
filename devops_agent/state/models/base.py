"""Base classes and primitives shared by all state models.

Design principles:
- All state models are immutable (frozen dataclasses).
- Mutation produces a new instance via ``evolve(**changes)``.
- Every persisted state carries a monotonic ``version`` for optimistic
  concurrency control (CAS pattern: UPDATE ... WHERE version = old_version).
- ``state_metadata`` carries audit fields (created_at, updated_at, node_name)
  and is always updated by the persistence layer, never by business logic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from typing import Any, TypeVar

T = TypeVar("T", bound="BaseState")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


@dataclass(frozen=True)
class StateVersion:
    """Monotonically increasing version counter for optimistic concurrency."""

    value: int = 0

    def increment(self) -> StateVersion:
        return StateVersion(value=self.value + 1)

    def __lt__(self, other: StateVersion) -> bool:  # noqa: D105
        return self.value < other.value

    def __le__(self, other: StateVersion) -> bool:
        return self.value <= other.value

    def __gt__(self, other: StateVersion) -> bool:
        return self.value > other.value

    def __ge__(self, other: StateVersion) -> bool:
        return self.value >= other.value


@dataclass(frozen=True)
class StateMetadata:
    """Audit metadata attached to every persisted state object.

    Populated and updated exclusively by the persistence layer; business
    logic must not modify these fields directly.
    """

    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    # The LangGraph node that produced this state version.
    producing_node: str | None = None
    # The LangGraph checkpoint_id that captured this version.
    checkpoint_id: str | None = None
    # Schema version for forward-compatible migration.
    schema_version: int = 1

    def touch(self, node: str | None = None) -> StateMetadata:
        """Return a copy with updated_at refreshed and node recorded."""
        return StateMetadata(
            created_at=self.created_at,
            updated_at=_utcnow(),
            producing_node=node or self.producing_node,
            checkpoint_id=self.checkpoint_id,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class BaseState:
    """Root base for all immutable state models.

    Subclasses must remain ``frozen=True``.  Use ``evolve(**changes)`` to
    produce updated instances; this mirrors the ``attrs.evolve`` pattern
    without the attrs dependency.
    """

    version: StateVersion = field(default_factory=StateVersion)
    metadata: StateMetadata = field(default_factory=StateMetadata)

    def evolve(self: T, **changes: Any) -> T:
        """Return a new instance with the supplied fields overridden.

        Always increments ``version`` and refreshes ``metadata.updated_at``
        unless those fields are explicitly included in ``changes``.
        """
        if "version" not in changes:
            changes["version"] = self.version.increment()
        if "metadata" not in changes:
            changes["metadata"] = self.metadata.touch()
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict representation for serialization entry points."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
