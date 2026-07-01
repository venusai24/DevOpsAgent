"""LangGraph-compatible checkpointer adapter.

Wraps our PostgresCheckpointRepository to satisfy the LangGraph
``BaseCheckpointSaver`` interface, enabling LangGraph's own ``graph.invoke()``
to persist and retrieve checkpoints using our persistence layer.

This adapter is the single integration point between LangGraph's internal
checkpoint machinery and our domain model.

Reference: LangGraph docs — PostgresSaver schema (``checkpoints``, ``writes``).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..interfaces.checkpoint_repository import CheckpointRepository
from ..serializers.investigation_serializer import InvestigationSerializer

logger = logging.getLogger(__name__)

_serializer = InvestigationSerializer()


class LangGraphCheckpointerAdapter:
    """Thin adapter that translates LangGraph checkpoint primitives to our domain types.

    LangGraph checkpointer interface (simplified):
        - ``put(config, checkpoint, metadata, new_versions)`` — write a new checkpoint
        - ``get_tuple(config)`` — get the latest or a specific checkpoint tuple
        - ``list(config, ...)`` — list checkpoint tuples for a thread

    This class implements the async variants.  The underlying storage is our
    ``PostgresCheckpointRepository``.
    """

    def __init__(self, checkpoint_repo: CheckpointRepository):
        self._repo = checkpoint_repo

    def _thread_id(self, config: dict[str, Any]) -> uuid.UUID:
        raw = config.get("configurable", {}).get("thread_id")
        if raw is None:
            raise ValueError(f"No thread_id in config: {config!r}")
        return uuid.UUID(str(raw))

    def _checkpoint_id(self, config: dict[str, Any]) -> uuid.UUID | None:
        raw = config.get("configurable", {}).get("checkpoint_id")
        if raw is None:
            return None
        return uuid.UUID(str(raw))

    async def aget_tuple(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Return the latest checkpoint tuple for the given config, or None."""
        thread_id = self._thread_id(config)
        checkpoint_id = self._checkpoint_id(config)
        cp = await self._repo.get_by_checkpoint_id(thread_id, checkpoint_id)             if checkpoint_id else await self._repo.get_latest(thread_id)
        if cp is None:
            return None
        return self._to_langgraph_tuple(cp)

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a LangGraph checkpoint and return the updated config."""
        thread_id = self._thread_id(config)
        parent_id = self._checkpoint_id(config)
        node_name = metadata.get("source", "unknown")

        # The channel_values dict IS the serialised InvestigationState payload
        # from LangGraph's perspective.  We deserialise it to verify the schema
        # and re-persist it through our typed checkpoint repository.
        channel_values = checkpoint.get("channel_values", {})
        try:
            inv_state = _serializer.deserialise(channel_values)
        except Exception:
            logger.warning(
                "LangGraph channel_values could not be deserialised as InvestigationState; "
                "storing raw payload for thread %s", thread_id
            )
            inv_state = None  # Store raw anyway — never drop a checkpoint

        cp = await self._repo.write_checkpoint(
            thread_id=thread_id,
            state=inv_state,  # type: ignore[arg-type]
            node_name=node_name,
            parent_checkpoint_id=parent_id,
            channel_versions=new_versions,
            pending_writes=checkpoint.get("pending_sends", []),
        )

        return {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_id": str(cp.checkpoint_id),
                "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
            }
        }

    async def alist(
        self,
        config: dict[str, Any],
        *,
        limit: int = 10,
        before: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield LangGraph checkpoint tuples in reverse-chronological order."""
        thread_id = self._thread_id(config)
        chain = await self._repo.list_chain(thread_id, limit=limit)
        for cp in chain:
            yield self._to_langgraph_tuple(cp)

    def _to_langgraph_tuple(self, cp) -> dict[str, Any]:
        """Convert a CheckpointState to LangGraph's checkpoint-tuple dict."""
        return {
            "config": {
                "configurable": {
                    "thread_id": str(cp.thread_id),
                    "checkpoint_id": str(cp.checkpoint_id),
                }
            },
            "checkpoint": {
                "id": str(cp.checkpoint_id),
                "channel_values": cp.payload or {},
                "channel_versions": cp.channel_versions,
                "versions_seen": cp.channel_versions,
                "pending_sends": cp.pending_writes,
            },
            "metadata": {
                "source": cp.producing_node,
                "step": cp.sequence_number,
                "writes": {},
                "parents": {
                    "": str(cp.parent_checkpoint_id) if cp.parent_checkpoint_id else ""
                },
            },
        }
