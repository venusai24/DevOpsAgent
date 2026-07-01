"""Concrete implementation of InvestigationRepository backed by PostgreSQL.

SQL conventions:
- investigation_registry table DDL: Phase 3 §3.1.2
- Overlap query uses GiST index on tstzrange (idx_inv_registry_time_range).
- Dedup claim uses SERIALIZABLE isolation (§3.2.2).
- All status transitions use optimistic concurrency (version CAS).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime

import asyncpg
from asyncpg.exceptions import SerializationError, UniqueViolationError

from ..adapters.postgres_adapter import PostgresAdapter
from ..enums.dedup_decision import DedupDecision
from ..enums.investigation_status import InvestigationStatus
from ..interfaces.investigation_repository import InvestigationRepository
from ..interfaces.repository import (
    ConcurrentModificationError,
    DedupConflictExhausted,
    EntityNotFoundError,
)
from ..models.base import StateMetadata, StateVersion
from ..models.execution_state import ExecutionState

logger = logging.getLogger(__name__)

MAX_DEDUP_RETRIES = 3


def _row_to_state(row: asyncpg.Record) -> ExecutionState:
    return ExecutionState(
        version=StateVersion(value=row["version"]),
        metadata=StateMetadata(
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        ),
        investigation_id=row["investigation_id"],
        cluster_id=row["cluster_id"],
        symptom_signature=row["symptom_signature"],
        time_range_start=row.get("time_range_start"),
        time_range_end=row.get("time_range_end"),
        status=InvestigationStatus(row["status"]),
        started_at=row["created_at"],
        completed_at=row.get("completed_at"),
        current_node=row.get("current_node"),
        dedup_decision=row.get("dedup_decision"),
        related_investigation_id=row.get("related_investigation_id"),
        topology_manifest_version=row.get("topology_manifest_version"),
        stage0_cache_key=row.get("stage0_cache_key"),
        confidence_level=row.get("confidence_level"),
        root_cause_cmdb_id=row.get("root_cause_cmdb_id"),
    )


class PostgresInvestigationRepository(InvestigationRepository):
    """Production implementation backed by asyncpg."""

    def __init__(self, adapter: PostgresAdapter):
        self._db = adapter

    async def get_by_id(self, entity_id: uuid.UUID) -> ExecutionState | None:
        row = await self._db.fetchrow(
            """
            SELECT investigation_id, cluster_id, symptom_signature,
                   lower(time_range) AS time_range_start,
                   upper(time_range) AS time_range_end,
                   status, dedup_decision, related_investigation_id,
                   topology_manifest_version, stage0_cache_key,
                   confidence_level, root_cause_cmdb_id, current_node,
                   created_at, updated_at, completed_at, version
            FROM investigation_registry
            WHERE investigation_id = $1
            """,
            entity_id,
        )
        return _row_to_state(row) if row else None

    async def save(self, entity: ExecutionState) -> ExecutionState:
        async with self._db.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO investigation_registry (
                    investigation_id, cluster_id, symptom_signature,
                    time_range, status, dedup_decision, related_investigation_id,
                    topology_manifest_version, stage0_cache_key,
                    current_node, confidence_level, root_cause_cmdb_id,
                    created_at, updated_at, version
                ) VALUES ($1, $2, $3, tstzrange($4, $5), $6, $7, $8,
                          $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (investigation_id) DO UPDATE
                    SET status = EXCLUDED.status,
                        current_node = EXCLUDED.current_node,
                        updated_at = EXCLUDED.updated_at,
                        version = EXCLUDED.version,
                        confidence_level = EXCLUDED.confidence_level,
                        root_cause_cmdb_id = EXCLUDED.root_cause_cmdb_id,
                        completed_at = EXCLUDED.completed_at
                WHERE investigation_registry.version = EXCLUDED.version - 1
                """,
                entity.investigation_id,
                entity.cluster_id,
                entity.symptom_signature,
                entity.time_range_start,
                entity.time_range_end,
                entity.status.value,
                entity.dedup_decision,
                entity.related_investigation_id,
                entity.topology_manifest_version,
                entity.stage0_cache_key,
                entity.current_node,
                entity.confidence_level,
                entity.root_cause_cmdb_id,
                entity.metadata.created_at,
                entity.metadata.updated_at,
                entity.version.value,
            )
        return entity

    async def delete(self, entity_id: uuid.UUID) -> None:
        await self._db.execute(
            "DELETE FROM investigation_registry WHERE investigation_id = $1",
            entity_id,
        )

    async def claim_with_dedup(
        self,
        investigation_id: uuid.UUID,
        cluster_id: str,
        symptom_signature: str,
        time_range_start: datetime,
        time_range_end: datetime,
    ) -> tuple[ExecutionState, DedupDecision, uuid.UUID | None]:
        """SERIALIZABLE transaction that atomically classifies and claims."""
        from ..enums.dedup_decision import DedupDecision as DD

        for attempt in range(MAX_DEDUP_RETRIES):
            try:
                async with self._db.serializable_transaction() as conn:
                    overlapping = await conn.fetch(
                        """
                        SELECT investigation_id, time_range, status,
                               symptom_signature, dedup_decision, stage0_cache_key
                        FROM investigation_registry
                        WHERE cluster_id = $1
                          AND time_range && tstzrange($2, $3)
                          AND status NOT IN ('resolved', 'superseded', 'duplicate_halted',
                                             'agent_failure', 'timeout')
                        FOR UPDATE
                        """,
                        cluster_id,
                        time_range_start,
                        time_range_end,
                    )

                    decision, related_id = _classify_dedup(
                        investigation_id=investigation_id,
                        symptom_signature=symptom_signature,
                        time_range_start=time_range_start,
                        time_range_end=time_range_end,
                        overlapping=overlapping,
                    )

                    row_status = (
                        InvestigationStatus.DUPLICATE_HALTED
                        if decision == DD.DUPLICATE
                        else InvestigationStatus.CLAIMED
                    )

                    await conn.execute(
                        """
                        INSERT INTO investigation_registry (
                            investigation_id, cluster_id, symptom_signature,
                            time_range, status, dedup_decision, related_investigation_id,
                            created_at, updated_at, version
                        ) VALUES ($1, $2, $3, tstzrange($4, $5), $6, $7, $8,
                                  now(), now(), 0)
                        ON CONFLICT (investigation_id) DO NOTHING
                        """,
                        investigation_id, cluster_id, symptom_signature,
                        time_range_start, time_range_end,
                        row_status.value, decision.value, related_id,
                    )

                    row = await conn.fetchrow(
                        "SELECT * FROM investigation_registry WHERE investigation_id = $1",
                        investigation_id,
                    )
                    state = _row_to_state(row)
                return state, decision, related_id

            except SerializationError:
                logger.warning(
                    "SERIALIZABLE dedup tx serialization_failure for %s (attempt %d/%d)",
                    investigation_id, attempt + 1, MAX_DEDUP_RETRIES,
                )
                continue

            except UniqueViolationError:
                # uq_inv_registry_live_duplicate backstop triggered.
                existing = await self._db.fetchrow(
                    """
                    SELECT investigation_id FROM investigation_registry
                    WHERE cluster_id = $1 AND symptom_signature = $2
                      AND status NOT IN ('resolved','superseded','duplicate_halted',
                                         'agent_failure','timeout')
                    """,
                    cluster_id, symptom_signature,
                )
                related_id = existing["investigation_id"] if existing else None
                state = await self.get_by_id(investigation_id)
                return state, DedupDecision.DUPLICATE, related_id

        raise DedupConflictExhausted(investigation_id)

    async def update_status(
        self,
        investigation_id: uuid.UUID,
        status: InvestigationStatus,
        current_node: str | None = None,
        expected_version: int | None = None,
    ) -> ExecutionState:
        async with self._db.transaction() as conn:
            if expected_version is not None:
                result = await conn.execute(
                    """
                    UPDATE investigation_registry
                    SET status = $1, current_node = COALESCE($2, current_node),
                        updated_at = now(), version = version + 1,
                        completed_at = CASE WHEN $1 IN (
                            'resolved','superseded','duplicate_halted',
                            'agent_failure','timeout'
                        ) THEN now() ELSE completed_at END
                    WHERE investigation_id = $3 AND version = $4
                    """,
                    status.value, current_node, investigation_id, expected_version,
                )
                if result == "UPDATE 0":
                    row = await conn.fetchrow(
                        "SELECT version FROM investigation_registry WHERE investigation_id = $1",
                        investigation_id,
                    )
                    actual = row["version"] if row else -1
                    raise ConcurrentModificationError(investigation_id, expected_version, actual)
            else:
                await conn.execute(
                    """
                    UPDATE investigation_registry
                    SET status = $1, current_node = COALESCE($2, current_node),
                        updated_at = now(), version = version + 1
                    WHERE investigation_id = $3
                    """,
                    status.value, current_node, investigation_id,
                )

        state = await self.get_by_id(investigation_id)
        if state is None:
            raise EntityNotFoundError("ExecutionState", investigation_id)
        return state

    async def find_overlapping(
        self,
        cluster_id: str,
        time_range_start: datetime,
        time_range_end: datetime,
        exclude_terminal: bool = True,
    ) -> Sequence[ExecutionState]:
        where_clause = (
            "AND status NOT IN ('resolved','superseded','duplicate_halted','agent_failure','timeout')"
            if exclude_terminal
            else ""
        )
        rows = await self._db.fetch(
            f"""
            SELECT investigation_id, cluster_id, symptom_signature,
                   lower(time_range) AS time_range_start,
                   upper(time_range) AS time_range_end,
                   status, dedup_decision, related_investigation_id,
                   topology_manifest_version, stage0_cache_key,
                   confidence_level, root_cause_cmdb_id, current_node,
                   created_at, updated_at, completed_at, version
            FROM investigation_registry
            WHERE cluster_id = $1 AND time_range && tstzrange($2, $3)
            {where_clause}
            ORDER BY created_at
            """,
            cluster_id, time_range_start, time_range_end,
        )
        return [_row_to_state(r) for r in rows]

    async def find_by_status(
        self,
        status: InvestigationStatus,
        cluster_id: str | None = None,
        limit: int = 100,
    ) -> Sequence[ExecutionState]:
        if cluster_id:
            rows = await self._db.fetch(
                """
                SELECT investigation_id, cluster_id, symptom_signature,
                       lower(time_range) AS time_range_start,
                       upper(time_range) AS time_range_end,
                       status, dedup_decision, related_investigation_id,
                       topology_manifest_version, stage0_cache_key,
                       confidence_level, root_cause_cmdb_id, current_node,
                       created_at, updated_at, completed_at, version
                FROM investigation_registry
                WHERE status = $1 AND cluster_id = $2
                ORDER BY created_at
                LIMIT $3
                """,
                status.value, cluster_id, limit,
            )
        else:
            rows = await self._db.fetch(
                """
                SELECT investigation_id, cluster_id, symptom_signature,
                       lower(time_range) AS time_range_start,
                       upper(time_range) AS time_range_end,
                       status, dedup_decision, related_investigation_id,
                       topology_manifest_version, stage0_cache_key,
                       confidence_level, root_cause_cmdb_id, current_node,
                       created_at, updated_at, completed_at, version
                FROM investigation_registry
                WHERE status = $1
                ORDER BY created_at
                LIMIT $2
                """,
                status.value, limit,
            )
        return [_row_to_state(r) for r in rows]

    async def mark_superseded(
        self,
        investigation_ids: Sequence[uuid.UUID],
        superseded_by: uuid.UUID,
    ) -> None:
        if not investigation_ids:
            return
        await self._db.execute(
            """
            UPDATE investigation_registry
            SET status = 'superseded', related_investigation_id = $1,
                updated_at = now(), version = version + 1
            WHERE investigation_id = ANY($2::uuid[])
              AND status NOT IN ('resolved','superseded','duplicate_halted')
            """,
            superseded_by,
            list(investigation_ids),
        )

    async def update_terminal_fields(
        self,
        investigation_id: uuid.UUID,
        confidence_level: str | None,
        root_cause_cmdb_id: str | None,
    ) -> None:
        await self._db.execute(
            """
            UPDATE investigation_registry
            SET confidence_level = $1, root_cause_cmdb_id = $2, updated_at = now()
            WHERE investigation_id = $3
            """,
            confidence_level, root_cause_cmdb_id, investigation_id,
        )

    async def find_stale_hitl(self, older_than: datetime) -> Sequence[ExecutionState]:
        rows = await self._db.fetch(
            """
            SELECT investigation_id, cluster_id, symptom_signature,
                   lower(time_range) AS time_range_start,
                   upper(time_range) AS time_range_end,
                   status, dedup_decision, related_investigation_id,
                   topology_manifest_version, stage0_cache_key,
                   confidence_level, root_cause_cmdb_id, current_node,
                   created_at, updated_at, completed_at, version
            FROM investigation_registry
            WHERE status = 'hitl_paused' AND updated_at < $1
            ORDER BY updated_at
            """,
            older_than,
        )
        return [_row_to_state(r) for r in rows]


def _classify_dedup(
    investigation_id: uuid.UUID,
    symptom_signature: str,
    time_range_start: datetime,
    time_range_end: datetime,
    overlapping: list,
) -> tuple[DedupDecision, uuid.UUID | None]:
    """Pure function: classify the dedup decision for a new investigation.

    No I/O.  Called inside the SERIALIZABLE transaction body.

    Rules (per ADR-001 Stage -1 spec):
    - DUPLICATE: exact match (same cluster, same symptom_signature) exists live.
    - SUBSET:    new window is fully within an existing window.
    - SUPERSET:  new window fully contains ALL overlapping windows.
    - PARTIAL_OVERLAP: otherwise overlapping.
    - NEW: no overlapping rows.
    """
    if not overlapping:
        return DedupDecision.NEW, None

    new_start = time_range_start
    new_end = time_range_end

    for row in overlapping:
        if row["symptom_signature"] == symptom_signature:
            return DedupDecision.DUPLICATE, row["investigation_id"]

    for row in overlapping:
        row_start = row["time_range"].lower if hasattr(row["time_range"], "lower") else new_start
        row_end = row["time_range"].upper if hasattr(row["time_range"], "upper") else new_end
        # SUBSET: new window is fully within an existing window.
        if row_start <= new_start and new_end <= row_end:
            return DedupDecision.SUBSET, row["investigation_id"]

    # SUPERSET: new window fully contains all overlapping windows.
    all_contained = all(
        new_start <= (row["time_range"].lower if hasattr(row["time_range"], "lower") else new_start)
        and (row["time_range"].upper if hasattr(row["time_range"], "upper") else new_end) <= new_end
        for row in overlapping
    )
    if all_contained:
        return DedupDecision.SUPERSET, overlapping[0]["investigation_id"]

    return DedupDecision.PARTIAL_OVERLAP, overlapping[0]["investigation_id"]
