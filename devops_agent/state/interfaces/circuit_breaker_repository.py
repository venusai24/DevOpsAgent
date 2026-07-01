"""Abstract repository for per-tool CircuitBreakerState persistence."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from ..models.circuit_breaker_state import CircuitBreakerState
from .repository import AsyncRepository


class CircuitBreakerRepository(AsyncRepository[CircuitBreakerState]):
    """Controls read/write for per-tool circuit breaker FSM state.

    One row per tool_name.  The in-memory CircuitBreakerRegistry consults this
    store to persist state changes, enabling shared state across worker processes
    when needed (Phase 4 §7.3).
    """

    @abstractmethod
    async def get_by_tool(self, tool_name: str) -> CircuitBreakerState | None:
        """Return the current CB state for a tool, or None."""

    @abstractmethod
    async def get_all(self) -> Sequence[CircuitBreakerState]:
        """Return all circuit breaker rows."""

    @abstractmethod
    async def upsert(self, state: CircuitBreakerState) -> CircuitBreakerState:
        """Upsert CB state, using optimistic concurrency on version."""

    @abstractmethod
    async def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED (for testing / admin ops)."""

    @abstractmethod
    async def get_by_id(self, entity_id) -> CircuitBreakerState | None:
        """Satisfy AsyncRepository[T] contract; delegates to get_by_tool via cb_id."""

    @abstractmethod
    async def save(self, entity: CircuitBreakerState) -> CircuitBreakerState:
        """Alias of upsert for compatibility with AsyncRepository contract."""

    @abstractmethod
    async def delete(self, entity_id) -> None:
        """Delete CB state by cb_id."""
