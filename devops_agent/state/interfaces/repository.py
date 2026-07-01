"""Generic async repository abstract base class."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncRepository(ABC, Generic[T]):
    """Base contract for all async repositories.

    - ``get_by_id`` returns None when the entity does not exist (not an error).
    - ``save`` is upsert-by-primary-key.
    - ``delete`` is idempotent (no error on missing entity).

    All methods are async; no blocking I/O is permitted.
    """

    @abstractmethod
    async def get_by_id(self, entity_id: uuid.UUID) -> T | None:
        ...

    @abstractmethod
    async def save(self, entity: T) -> T:
        ...

    @abstractmethod
    async def delete(self, entity_id: uuid.UUID) -> None:
        ...


class ConcurrentModificationError(Exception):
    """Raised when an optimistic-concurrency CAS update finds a version mismatch."""

    def __init__(self, entity_id: uuid.UUID, expected_version: int, actual_version: int):
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Concurrent modification detected for {entity_id}: "
            f"expected version {expected_version}, found {actual_version}"
        )


class EntityNotFoundError(Exception):
    """Raised when a required entity is not found (as opposed to Optional returns)."""

    def __init__(self, entity_type: str, entity_id: uuid.UUID):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} not found: {entity_id}")


class DedupConflictExhausted(Exception):
    """Raised when the SERIALIZABLE dedup transaction exhausts retries."""

    def __init__(self, investigation_id: uuid.UUID):
        self.investigation_id = investigation_id
        super().__init__(
            f"Dedup SERIALIZABLE transaction retries exhausted for investigation {investigation_id}"
        )
