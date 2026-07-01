"""Abstract repository for the LangGraph Checkpoint store (data plane)."""

from __future__ import annotations

import uuid
from abc import abstractmethod
from collections.abc import Sequence

from ..models.checkpoint_state import CheckpointState
from ..models.investigation_state import InvestigationState
from .repository import AsyncRepository


class CheckpointRepository(AsyncRepository[CheckpointState]):
    """Controls read/write access to the LangGraph checkpoint tables.

    The checkpoint table schema follows PostgresSaver's conventions; this
    interface layers our domain types on top.
    """

    @abstractmethod
    async def write_checkpoint(
        self,
        thread_id: uuid.UUID,
        state: InvestigationState,
        node_name: str,
        parent_checkpoint_id: uuid.UUID | None = None,
        channel_versions: dict | None = None,
        pending_writes: list | None = None,
    ) -> CheckpointState:
        """Serialise and persist a new checkpoint for the given thread.

        Atomically marks the previous latest checkpoint as historical and
        writes the new one as latest.  Raises no error if parent_checkpoint_id
        is None (first checkpoint in the chain).
        """

    @abstractmethod
    async def get_latest(self, thread_id: uuid.UUID) -> CheckpointState | None:
        """Return the most recent checkpoint for a thread, or None."""

    @abstractmethod
    async def get_by_checkpoint_id(
        self, thread_id: uuid.UUID, checkpoint_id: uuid.UUID
    ) -> CheckpointState | None:
        """Return a specific historical checkpoint by its ID."""

    @abstractmethod
    async def list_chain(
        self,
        thread_id: uuid.UUID,
        limit: int = 50,
    ) -> Sequence[CheckpointState]:
        """Return the checkpoint chain in reverse-chronological order."""

    @abstractmethod
    async def load_state(
        self,
        thread_id: uuid.UUID,
        checkpoint_id: uuid.UUID | None = None,
    ) -> InvestigationState | None:
        """Deserialise and return InvestigationState from a checkpoint.

        Loads the latest checkpoint when ``checkpoint_id`` is None.
        """

    @abstractmethod
    async def delete_thread(self, thread_id: uuid.UUID) -> int:
        """Delete all checkpoint rows for a thread.  Returns count deleted.

        Called by the retention job after an investigation reaches terminal
        status + grace period (Phase 3 §3.3.5).
        """

    @abstractmethod
    async def mark_checkpoint_paused(self, checkpoint_id: uuid.UUID) -> None:
        """Update checkpoint_status to 'hitl_paused' for the given checkpoint."""

    @abstractmethod
    async def mark_checkpoint_resumed(self, checkpoint_id: uuid.UUID) -> None:
        """Update checkpoint_status back to 'active' after HITL resume."""
