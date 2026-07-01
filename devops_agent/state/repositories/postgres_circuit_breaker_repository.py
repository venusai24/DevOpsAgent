"""Concrete CircuitBreakerRepository backed by PostgreSQL."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

import asyncpg

from ..adapters.postgres_adapter import PostgresAdapter
from ..enums.circuit_breaker_status import CircuitBreakerStatus
from ..interfaces.circuit_breaker_repository import CircuitBreakerRepository
from ..models.base import StateMetadata, StateVersion
from ..models.circuit_breaker_state import CircuitBreakerState

logger = logging.getLogger(__name__)


def _row_to_state(row: asyncpg.Record) -> CircuitBreakerState:
    return CircuitBreakerState(
        version=StateVersion(value=row.get("version", 0)),
        metadata=StateMetadata(
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        ),
        cb_id=row["cb_id"],
        tool_name=row["tool_name"],
        status=CircuitBreakerStatus(row["status"]),
        failure_count=row.get("failure_count", 0),
        failure_threshold=row.get("failure_threshold", 5),
        opened_at=row.get("opened_at"),
        half_open_since=row.get("half_open_since"),
        open_timeout_s=row.get("open_timeout_s", 120),
        total_open_transitions=row.get("total_open_transitions", 0),
        total_close_transitions=row.get("total_close_transitions", 0),
        last_failure_at=row.get("last_failure_at"),
        last_success_at=row.get("last_success_at"),
    )


class PostgresCircuitBreakerRepository(CircuitBreakerRepository):
    """Production implementation.  One row per tool_name."""

    def __init__(self, adapter: PostgresAdapter):
        self._db = adapter

    async def get_by_id(self, entity_id: uuid.UUID) -> CircuitBreakerState | None:
        row = await self._db.fetchrow(
            "SELECT * FROM circuit_breaker_state WHERE cb_id = $1", entity_id
        )
        return _row_to_state(row) if row else None

    async def get_by_tool(self, tool_name: str) -> CircuitBreakerState | None:
        row = await self._db.fetchrow(
            "SELECT * FROM circuit_breaker_state WHERE tool_name = $1", tool_name
        )
        return _row_to_state(row) if row else None

    async def get_all(self) -> Sequence[CircuitBreakerState]:
        rows = await self._db.fetch(
            "SELECT * FROM circuit_breaker_state ORDER BY tool_name"
        )
        return [_row_to_state(r) for r in rows]

    async def upsert(self, state: CircuitBreakerState) -> CircuitBreakerState:
        async with self._db.transaction() as conn:
            result = await conn.execute(
                """
                INSERT INTO circuit_breaker_state (
                    cb_id, tool_name, status, failure_count, failure_threshold,
                    opened_at, half_open_since, open_timeout_s,
                    total_open_transitions, total_close_transitions,
                    last_failure_at, last_success_at,
                    created_at, updated_at, version
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,now(),now(),$13)
                ON CONFLICT (tool_name) DO UPDATE
                    SET status = EXCLUDED.status,
                        failure_count = EXCLUDED.failure_count,
                        opened_at = EXCLUDED.opened_at,
                        half_open_since = EXCLUDED.half_open_since,
                        total_open_transitions = EXCLUDED.total_open_transitions,
                        total_close_transitions = EXCLUDED.total_close_transitions,
                        last_failure_at = EXCLUDED.last_failure_at,
                        last_success_at = EXCLUDED.last_success_at,
                        updated_at = now(),
                        version = circuit_breaker_state.version + 1
                    WHERE circuit_breaker_state.version = $13 - 1
                """,
                state.cb_id, state.tool_name, state.status.value,
                state.failure_count, state.failure_threshold,
                state.opened_at, state.half_open_since, state.open_timeout_s,
                state.total_open_transitions, state.total_close_transitions,
                state.last_failure_at, state.last_success_at,
                state.version.value,
            )
        return await self.get_by_tool(state.tool_name) or state

    async def save(self, entity: CircuitBreakerState) -> CircuitBreakerState:
        return await self.upsert(entity)

    async def delete(self, entity_id: uuid.UUID) -> None:
        await self._db.execute(
            "DELETE FROM circuit_breaker_state WHERE cb_id = $1", entity_id
        )

    async def reset_all(self) -> None:
        await self._db.execute(
            """
            UPDATE circuit_breaker_state
            SET status = 'closed', failure_count = 0, opened_at = NULL,
                half_open_since = NULL, updated_at = now(), version = version + 1
            """
        )
        logger.info("All circuit breakers reset to CLOSED")
