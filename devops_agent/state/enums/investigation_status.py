"""Investigation lifecycle status enumeration.

Mirrors the investigation_status PostgreSQL ENUM defined in
StateManagement&PersistenceLayer.md §3.1.2.
"""

from enum import StrEnum


class InvestigationStatus(StrEnum):
    """Every status that an investigation_registry row can carry.

    Terminal statuses: RESOLVED, SUPERSEDED, DUPLICATE_HALTED.
    The partial index ``idx_inv_registry_live_by_cluster`` and the unique
    constraint ``uq_inv_registry_live_duplicate`` both filter on
    ``status NOT IN ('resolved', 'superseded', 'duplicate_halted')``.
    """

    # Row reserved; dedup transaction committed, graph not yet routed.
    CLAIMED = "claimed"

    # Stage 0 in flight (NEW / SUPERSET / PARTIAL_OVERLAP routes here).
    PENDING_CONTEXT = "pending_context"

    # Triage / RCA in progress.
    ACTIVE = "active"

    # Interrupted, awaiting human input via HITL.
    HITL_PAUSED = "hitl_paused"

    # Terminal: report delivered.
    RESOLVED = "resolved"

    # Terminal: subsumed by a broader (SUPERSET) investigation.
    SUPERSEDED = "superseded"

    # Terminal: exact duplicate; never independently run.
    DUPLICATE_HALTED = "duplicate_halted"

    # --- Phase 4 additions --------------------------------------------------
    # Unrecoverable LLM / tool failure after retry exhaustion.
    AGENT_FAILURE = "agent_failure"

    # Per-node or global wall-clock timeout triggered.
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        """Return True when the investigation has reached a final state."""
        return self in (
            InvestigationStatus.RESOLVED,
            InvestigationStatus.SUPERSEDED,
            InvestigationStatus.DUPLICATE_HALTED,
            InvestigationStatus.AGENT_FAILURE,
            InvestigationStatus.TIMEOUT,
        )

    @property
    def is_live(self) -> bool:
        """Return True for statuses included in the partial live index."""
        return not self.is_terminal
