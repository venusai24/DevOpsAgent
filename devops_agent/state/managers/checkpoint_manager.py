"""CheckpointManager — orchestrates pause, resume, restart, rollback, replay.

This is the primary entry point for all checkpoint lifecycle operations.
It composes the CheckpointRepository, InvestigationRepository, and
InvestigationSerializer to implement the operations listed in the spec.

No LangGraph API is called here — the manager works at the domain level.
LangGraph integration happens via LangGraphCheckpointerAdapter.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from ..enums.investigation_status import InvestigationStatus
from ..interfaces.checkpoint_repository import CheckpointRepository
from ..interfaces.investigation_repository import InvestigationRepository
from ..interfaces.repository import EntityNotFoundError
from ..models.checkpoint_state import CheckpointState
from ..models.context_snapshot import ContextSnapshot
from ..models.investigation_state import InvestigationState

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CheckpointManager:
    """High-level checkpoint lifecycle manager.

    Supported operations:
        - pause(investigation_id): persist HITL pause; update status.
        - resume(investigation_id, human_input): load latest checkpoint, inject input.
        - restart(investigation_id): roll back to the first checkpoint (cold restart).
        - rollback(investigation_id, checkpoint_id): rewind to a specific checkpoint.
        - replay(investigation_id, from_checkpoint_id): walk the chain from a snapshot.
        - snapshot(investigation_id): build a ContextSnapshot without loading payload.
    """

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        investigation_repo: InvestigationRepository,
    ):
        self._cp = checkpoint_repo
        self._inv = investigation_repo

    # ── pause ──────────────────────────────────────────────────────────────────

    async def pause(self, investigation_id: uuid.UUID, trigger: str) -> CheckpointState:
        """Mark the investigation and its latest checkpoint as HITL paused.

        Idempotent: if already paused, returns the existing state.
        """
        cp = await self._cp.get_latest(investigation_id)
        if cp is None:
            raise EntityNotFoundError("CheckpointState", investigation_id)
        if cp.checkpoint_status == "hitl_paused":
            return cp
        await self._cp.mark_checkpoint_paused(cp.checkpoint_id)
        await self._inv.update_status(investigation_id, InvestigationStatus.HITL_PAUSED)
        logger.info(
            "Investigation %s paused at checkpoint %s (trigger=%s)",
            investigation_id, cp.checkpoint_id, trigger,
        )
        return await self._cp.get_latest(investigation_id) or cp

    # ── resume ─────────────────────────────────────────────────────────────────

    async def resume(
        self,
        investigation_id: uuid.UUID,
        human_input: dict | None = None,
    ) -> tuple[CheckpointState, InvestigationState | None]:
        """Hydrate the latest checkpoint and mark the investigation as ACTIVE.

        Returns (checkpoint_metadata, hydrated_InvestigationState).
        The caller (ResumeManager) is responsible for injecting human_input
        into the state before handing it back to LangGraph.
        """
        cp = await self._cp.get_latest(investigation_id)
        if cp is None:
            raise EntityNotFoundError("CheckpointState", investigation_id)

        state = await self._cp.load_state(investigation_id)
        await self._cp.mark_checkpoint_resumed(cp.checkpoint_id)
        await self._inv.update_status(investigation_id, InvestigationStatus.ACTIVE)

        logger.info(
            "Investigation %s resumed from checkpoint %s", investigation_id, cp.checkpoint_id
        )
        return cp, state

    # ── restart ────────────────────────────────────────────────────────────────

    async def restart(self, investigation_id: uuid.UUID) -> InvestigationState | None:
        """Roll the investigation back to its first checkpoint.

        Use for cold restarts after a fatal failure.  The caller must re-invoke
        graph.invoke() with the returned state as the starting point.
        """
        chain = await self._cp.list_chain(investigation_id, limit=1000)
        if not chain:
            raise EntityNotFoundError("CheckpointState", investigation_id)
        first = chain[-1]  # chain is reverse-chron; last item = oldest
        state = await self._cp.load_state(investigation_id, first.checkpoint_id)
        await self._inv.update_status(investigation_id, InvestigationStatus.CLAIMED)
        logger.info(
            "Investigation %s restarted from first checkpoint %s",
            investigation_id, first.checkpoint_id,
        )
        return state

    # ── rollback ───────────────────────────────────────────────────────────────

    async def rollback(
        self,
        investigation_id: uuid.UUID,
        to_checkpoint_id: uuid.UUID,
    ) -> InvestigationState | None:
        """Rewind to a specific checkpoint snapshot.

        All checkpoints newer than ``to_checkpoint_id`` are NOT deleted (they
        form the historical audit trail) but the target checkpoint is promoted
        to is_latest=True so that the next write builds on it.
        """
        cp = await self._cp.get_by_checkpoint_id(investigation_id, to_checkpoint_id)
        if cp is None:
            raise EntityNotFoundError("CheckpointState", to_checkpoint_id)

        # Demote all checkpoints with seq > target.
        chain = await self._cp.list_chain(investigation_id, limit=1000)
        for historical in chain:
            if historical.checkpoint_id != to_checkpoint_id and historical.is_latest:
                await self._cp.save(historical.as_historical())

        # Promote the target.
        promoted = cp.evolve(is_latest=True, checkpoint_status="active")
        await self._cp.save(promoted)

        state = await self._cp.load_state(investigation_id, to_checkpoint_id)
        await self._inv.update_status(investigation_id, InvestigationStatus.ACTIVE)
        logger.info(
            "Investigation %s rolled back to checkpoint %s",
            investigation_id, to_checkpoint_id,
        )
        return state

    # ── replay ─────────────────────────────────────────────────────────────────

    async def replay(
        self,
        investigation_id: uuid.UUID,
        from_checkpoint_id: uuid.UUID | None = None,
    ) -> Sequence[tuple[CheckpointState, InvestigationState | None]]:
        """Walk the checkpoint chain from a given point, yielding (metadata, state) pairs.

        Yields in chronological order (oldest first).
        Used for debugging and audit.  Does NOT modify any state.
        """
        chain = await self._cp.list_chain(investigation_id, limit=1000)
        # chain is reverse-chron; reverse to chronological.
        chain = list(reversed(chain))

        if from_checkpoint_id:
            idx = next(
                (i for i, cp in enumerate(chain) if cp.checkpoint_id == from_checkpoint_id),
                None,
            )
            if idx is None:
                raise EntityNotFoundError("CheckpointState", from_checkpoint_id)
            chain = chain[idx:]

        result = []
        for cp in chain:
            state = await self._cp.load_state(investigation_id, cp.checkpoint_id)
            result.append((cp, state))
        return result

    # ── snapshot ───────────────────────────────────────────────────────────────

    async def snapshot(self, investigation_id: uuid.UUID) -> ContextSnapshot | None:
        """Build a ContextSnapshot without loading the full checkpoint payload.

        Used by the ResumeManager and the Stage 6 staleness check (§3.5.2).
        """
        cp = await self._cp.get_latest(investigation_id)
        if cp is None:
            return None
        reg = await self._inv.get_by_id(investigation_id)

        state = await self._cp.load_state(investigation_id)
        guardrails_summary = state.guardrails if state else None

        return ContextSnapshot(
            investigation_id=investigation_id,
            checkpoint_id=cp.checkpoint_id,
            status=reg.status if reg else InvestigationStatus.CLAIMED,
            producing_node=cp.producing_node,
            last_completed_stage="",  # Populated by caller from state if needed.
            topology_manifest_version=reg.topology_manifest_version if reg else None,
            forced_exit_triggered=guardrails_summary.forced_exit_triggered if guardrails_summary else False,
            trace_degradation_active=guardrails_summary.degradation_path_active if guardrails_summary else False,
            global_timeout_triggered=guardrails_summary.global_timeout_triggered if guardrails_summary else False,
            payload_schema_version=cp.payload_schema_version,
        )
