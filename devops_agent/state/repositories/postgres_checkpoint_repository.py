"""Concrete CheckpointRepository backed by PostgreSQL.

Table: checkpoints (separate from investigation_registry).
The payload column (JSONB) stores the full serialised InvestigationState.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any

import asyncpg

from ..adapters.postgres_adapter import PostgresAdapter
from ..interfaces.checkpoint_repository import CheckpointRepository
from ..models.base import StateMetadata, StateVersion
from ..models.checkpoint_state import CheckpointState
from ..models.investigation_state import InvestigationState
from ..serializers.investigation_serializer import InvestigationSerializer

logger = logging.getLogger(__name__)

_ser = InvestigationSerializer()


def _row_to_cp(row: asyncpg.Record) -> CheckpointState:
    raw_payload = row.get("payload")
    payload: dict[str, Any] | None = (
        json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
    )
    return CheckpointState(
        version=StateVersion(value=row.get("version", 0)),
        metadata=StateMetadata(
            created_at=row["created_at"],
            updated_at=row["created_at"],
            producing_node=row.get("producing_node"),
            checkpoint_id=str(row["checkpoint_id"]),
        ),
        checkpoint_id=row["checkpoint_id"],
        thread_id=row["thread_id"],
        parent_checkpoint_id=row.get("parent_checkpoint_id"),
        producing_node=row.get("producing_node", ""),
        sequence_number=row.get("sequence_number", 0),
        created_at=row["created_at"],
        payload=payload,
        payload_schema_version=row.get("payload_schema_version", 1),
        is_latest=row.get("is_latest", False),
        checkpoint_status=row.get("checkpoint_status", "active"),
        channel_versions=json.loads(row.get("channel_versions") or "{}"),
        pending_writes=json.loads(row.get("pending_writes") or "[]"),
    )


class PostgresCheckpointRepository(CheckpointRepository):
    """Production implementation backed by asyncpg."""

    def __init__(self, adapter: PostgresAdapter):
        self._db = adapter

    async def get_by_id(self, entity_id: uuid.UUID) -> CheckpointState | None:
        row = await self._db.fetchrow(
            "SELECT * FROM checkpoints WHERE checkpoint_id = $1", entity_id
        )
        return _row_to_cp(row) if row else None

    async def save(self, entity: CheckpointState) -> CheckpointState:
        payload_json = json.dumps(entity.payload) if entity.payload else None
        await self._db.execute(
            """
            INSERT INTO checkpoints (
                checkpoint_id, thread_id, parent_checkpoint_id, producing_node,
                sequence_number, created_at, payload, payload_schema_version,
                is_latest, checkpoint_status, channel_versions, pending_writes, version
            ) VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11::jsonb,$12::jsonb,$13)
            ON CONFLICT (checkpoint_id) DO UPDATE
                SET is_latest = EXCLUDED.is_latest,
                    checkpoint_status = EXCLUDED.checkpoint_status,
                    payload = EXCLUDED.payload,
                    version = EXCLUDED.version
            """,
            entity.checkpoint_id, entity.thread_id, entity.parent_checkpoint_id,
            entity.producing_node, entity.sequence_number, entity.created_at,
            payload_json,
            entity.payload_schema_version, entity.is_latest,
            entity.checkpoint_status,
            json.dumps(entity.channel_versions),
            json.dumps(entity.pending_writes),
            entity.version.value,
        )
        return entity

    async def delete(self, entity_id: uuid.UUID) -> None:
        await self._db.execute(
            "DELETE FROM checkpoints WHERE checkpoint_id = $1", entity_id
        )

    async def write_checkpoint(
        self,
        thread_id: uuid.UUID,
        state: InvestigationState,
        node_name: str,
        parent_checkpoint_id: uuid.UUID | None = None,
        channel_versions: dict | None = None,
        pending_writes: list | None = None,
    ) -> CheckpointState:
        checkpoint_id = uuid.uuid4()
        payload: dict | None = None
        if state is not None:
            try:
                payload = _ser.serialise(state)
            except Exception as exc:
                logger.error(
                    "Failed to serialise InvestigationState for thread %s: %s", thread_id, exc
                )

        # Determine sequence_number.
        latest = await self.get_latest(thread_id)
        seq = (latest.sequence_number + 1) if latest else 0

        async with self._db.transaction() as conn:
            # Mark previous latest as historical.
            if latest:
                await conn.execute(
                    "UPDATE checkpoints SET is_latest = FALSE WHERE thread_id = $1 AND is_latest = TRUE",
                    thread_id,
                )
            # Insert new checkpoint.
            payload_json = json.dumps(payload) if payload else None
            await conn.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, thread_id, parent_checkpoint_id, producing_node,
                    sequence_number, created_at, payload, payload_schema_version,
                    is_latest, checkpoint_status, channel_versions, pending_writes, version
                ) VALUES ($1,$2,$3,$4,$5,now(),$6::jsonb,$7,$8,$9,$10::jsonb,$11::jsonb,0)
                """,
                checkpoint_id, thread_id, parent_checkpoint_id, node_name,
                seq, payload_json, _ser.SCHEMA_VERSION,
                True, "active",
                json.dumps(channel_versions or {}),
                json.dumps(pending_writes or []),
            )

        row = await self._db.fetchrow(
            "SELECT * FROM checkpoints WHERE checkpoint_id = $1", checkpoint_id
        )
        return _row_to_cp(row)

    async def get_latest(self, thread_id: uuid.UUID) -> CheckpointState | None:
        row = await self._db.fetchrow(
            "SELECT * FROM checkpoints WHERE thread_id = $1 AND is_latest = TRUE",
            thread_id,
        )
        return _row_to_cp(row) if row else None

    async def get_by_checkpoint_id(
        self, thread_id: uuid.UUID, checkpoint_id: uuid.UUID
    ) -> CheckpointState | None:
        row = await self._db.fetchrow(
            "SELECT * FROM checkpoints WHERE thread_id = $1 AND checkpoint_id = $2",
            thread_id, checkpoint_id,
        )
        return _row_to_cp(row) if row else None

    async def list_chain(
        self,
        thread_id: uuid.UUID,
        limit: int = 50,
    ) -> Sequence[CheckpointState]:
        rows = await self._db.fetch(
            """
            SELECT * FROM checkpoints WHERE thread_id = $1
            ORDER BY sequence_number DESC LIMIT $2
            """,
            thread_id, limit,
        )
        return [_row_to_cp(r) for r in rows]

    async def load_state(
        self,
        thread_id: uuid.UUID,
        checkpoint_id: uuid.UUID | None = None,
    ) -> InvestigationState | None:
        cp = (
            await self.get_by_checkpoint_id(thread_id, checkpoint_id)
            if checkpoint_id
            else await self.get_latest(thread_id)
        )
        if cp is None or cp.payload is None:
            return None
        try:
            return _ser.deserialise(cp.payload)
        except Exception as exc:
            logger.error(
                "Failed to deserialise checkpoint %s for thread %s: %s",
                cp.checkpoint_id, thread_id, exc,
            )
            return None

    async def delete_thread(self, thread_id: uuid.UUID) -> int:
        result = await self._db.execute(
            "DELETE FROM checkpoints WHERE thread_id = $1", thread_id
        )
        try:
            count = int(result.split()[-1])
        except (ValueError, IndexError):
            count = 0
        logger.info("Deleted %d checkpoints for thread %s", count, thread_id)
        return count

    async def mark_checkpoint_paused(self, checkpoint_id: uuid.UUID) -> None:
        await self._db.execute(
            "UPDATE checkpoints SET checkpoint_status = 'hitl_paused' WHERE checkpoint_id = $1",
            checkpoint_id,
        )

    async def mark_checkpoint_resumed(self, checkpoint_id: uuid.UUID) -> None:
        await self._db.execute(
            "UPDATE checkpoints SET checkpoint_status = 'active' WHERE checkpoint_id = $1",
            checkpoint_id,
        )
