"""ResumeManager — handles HITL resume hydration and topology staleness check.

Spec: Phase 3 §3.3.3 (hydration) and §3.5.2 (topology staleness at HITL-resume).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from ..interfaces.investigation_repository import InvestigationRepository
from ..managers.checkpoint_manager import CheckpointManager
from ..models.context_snapshot import ContextSnapshot
from ..models.investigation_state import InvestigationState

logger = logging.getLogger(__name__)


class TopologyManifestMismatchError(Exception):
    """Raised when the investigation's manifest version differs from the current version."""

    def __init__(
        self,
        investigation_id: uuid.UUID,
        stored_version: str,
        current_version: str,
    ):
        self.investigation_id = investigation_id
        self.stored_version = stored_version
        self.current_version = current_version
        super().__init__(
            f"Topology manifest version mismatch for {investigation_id}: "
            f"stored={stored_version}, current={current_version}"
        )


class ResumeManager:
    """Orchestrates the full HITL resume flow.

    1. Validate the investigation is resumable (is_resumable check).
    2. Check topology manifest version (§3.5.2).
    3. Hydrate InvestigationState from the latest checkpoint.
    4. Inject the human_input response.
    5. Transition status to ACTIVE.
    6. Return the hydrated state to the caller for graph.invoke().

    The caller is responsible for passing the hydrated state back to LangGraph.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        investigation_repo: InvestigationRepository,
        current_manifest_version_fn: Any | None = None,
    ):
        """
        Args:
            checkpoint_manager: Provides pause/resume/hydration.
            investigation_repo: Provides registry row reads.
            current_manifest_version_fn: Optional async callable that returns
                the current topology manifest version for a cluster_id.
                Signature: async (cluster_id: str) -> str
        """
        self._ckpt = checkpoint_manager
        self._inv = investigation_repo
        self._get_current_manifest = current_manifest_version_fn

    async def resume(
        self,
        investigation_id: uuid.UUID,
        human_input: dict[str, Any],
    ) -> InvestigationState:
        """Full HITL resume with staleness check and state hydration.

        Returns the hydrated InvestigationState with hitl_response injected.
        Raises:
            EntityNotFoundError: investigation or checkpoint does not exist.
            ValueError: investigation is not in a resumable state.
            TopologyManifestMismatchError: manifest version differs (caller
                must decide whether to re-fetch topology or escalate).
        """
        # 1. Validate resumability.
        snapshot = await self._ckpt.snapshot(investigation_id)
        if snapshot is None:
            from ..interfaces.repository import EntityNotFoundError
            raise EntityNotFoundError("CheckpointState", investigation_id)

        if not snapshot.is_resumable:
            raise ValueError(
                f"Investigation {investigation_id} is not in a resumable state "
                f"(status={snapshot.status})"
            )

        # 2. Topology manifest staleness check (§3.5.2).
        await self._check_topology_staleness(snapshot)

        # 3. Hydrate InvestigationState.
        _cp, state = await self._ckpt.resume(investigation_id)

        if state is None:
            from ..interfaces.repository import EntityNotFoundError
            raise EntityNotFoundError("InvestigationState", investigation_id)

        # 4. Inject human_input.
        state = state.evolve(hitl_response=human_input)

        # 5. Record the HITL resume in timeout tracking.
        if state.timeouts is not None:
            ts = state.timeouts.resume_hitl(at=datetime.now(UTC))
            state = state.with_timeouts(ts)

        logger.info("Investigation %s HITL resume hydration complete", investigation_id)
        return state

    async def _check_topology_staleness(self, snapshot: ContextSnapshot) -> None:
        """If a manifest-version getter is configured, compare versions.

        Only runs the narrow re-fetch check; never triggers a full Stage 0 re-run.
        """
        if self._get_current_manifest is None:
            return
        if snapshot.topology_manifest_version is None:
            return

        reg = await self._inv.get_by_id(snapshot.investigation_id)
        if reg is None:
            return

        try:
            current_version = await self._get_current_manifest(reg.cluster_id)
        except Exception as exc:
            logger.warning(
                "Failed to fetch current manifest version for cluster %s: %s",
                reg.cluster_id, exc,
            )
            return

        if current_version != snapshot.topology_manifest_version:
            logger.warning(
                "Topology manifest version mismatch for investigation %s "
                "(stored=%s, current=%s) — narrow re-fetch required",
                snapshot.investigation_id,
                snapshot.topology_manifest_version,
                current_version,
            )
            raise TopologyManifestMismatchError(
                investigation_id=snapshot.investigation_id,
                stored_version=snapshot.topology_manifest_version,
                current_version=current_version,
            )
