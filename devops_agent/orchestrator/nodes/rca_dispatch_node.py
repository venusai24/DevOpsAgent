"""RCA Fan-Out Dispatcher.

Uses LangGraph's Send API to spawn one independent RCA branch per affected component
identified by the Triage Agent. Each branch receives its own isolated budget counters
and message history so branches never interfere with each other.

If triage produces no candidates (e.g. loop-prevention fallback), a single un-scoped
branch is dispatched — preserving the original sequential behaviour.
"""

from __future__ import annotations

from langgraph.types import Send

from ..state import GLOBAL_FEEDBACK_KEY, InvestigationState

# Keys that must NOT be written back by parallel branches — they are singleton
# (no Annotated reducer) and writing them from multiple concurrent branches
# will trigger LangGraph's InvalidUpdateError.
# Branches can still READ these values from the state they receive, but must
# not include them in their return dicts.
_PARALLEL_UNSAFE_KEYS: frozenset[str] = frozenset({
    "investigation_id",
    "time_range",
    "investigation_state",
    "current_trace_run_id",
    "current_node",
    "blast_radius",
    "blast_radius_qualifier",
    "T0",
    "ranked_hypotheses",
    "investigation_cluster",
    "affected_component_candidates",
    "baseline_registry_ref",
    "component_kpi_map",
    "declared_topology_graph",
    "discovered_topology_graph",
    "app_stats_path",
    "metrics_path",
    "logs_path",
    "traces_path",
    "dedup_decision",
    "reuse_investigation_id",
    "stage0_artifacts_available",
    "concurrent_investigation_ids",
    "component_registry",
    "tc_to_operation_map",
    "stack_kpi_map",
    "stage_0_gaps",
    "symptom_pattern",
    "anomalous_tc_values",
    "T0_sources",
    "leading_indicators",
    "incident_state",
    "concurrent_incident_clusters",
    "boundary_ambiguous_components",
})

# ──────────────────────────────────────────────────────────────────────────────
# Per-branch budget constants
# ──────────────────────────────────────────────────────────────────────────────

PER_BRANCH_TOTAL_BUDGET: int = 30  # Max tool invocations per branch
PER_BRANCH_HYP_BUDGET: int = 10     # Max tool invocations per hypothesis per branch


def _resolve_branch_feedback(
    critic_feedback: dict[str, str] | str | None,
    branch_id: str,
) -> str | None:
    """Resolve top-level critic_feedback_for_rca to a single branch's view of it.

    Handles three shapes:
      - None: no feedback yet.
      - str: a legacy value persisted by a checkpoint from before this field
        became a dict (a real investigation can pause at a HITL node and
        resume after a deploy). Broadcast unchanged, matching prior behaviour.
      - dict: combine this branch's own bucket with the GLOBAL_FEEDBACK_KEY
        bucket (unattributable feedback, e.g. a topological mismatch with no
        backing evidence_id), so nothing is silently lost.
    """
    if critic_feedback is None:
        return None
    if isinstance(critic_feedback, str):
        return critic_feedback

    parts = [
        critic_feedback[key]
        for key in (branch_id, GLOBAL_FEEDBACK_KEY)
        if critic_feedback.get(key)
    ]
    return "\n\n".join(parts) if parts else None


def dispatch_rca_fan_out(state: InvestigationState) -> list[Send]:
    """Return one Send per affected component, each with isolated branch state.

    LangGraph will execute all returned Sends concurrently. When every branch
    finishes, control passes to the ``merge_rca`` node, which consolidates results
    into the shared InvestigationState.
    """
    candidates: list[str] = state.get("affected_component_candidates", [])

    # Critic feedback (or the deterministic auto-correction path's equivalent)
    # from a previous RCA pass, keyed by branch_id + a global fallback bucket.
    # Resolved per-branch below so each Send only ever sees a plain str|None —
    # rca_llm_node is unaware this was ever a dict.
    critic_feedback: dict[str, str] | str | None = state.get("critic_feedback_for_rca")

    # Build the base state that is passed INTO each branch.
    # We pass the full state so branches can read everything they need.
    # However, we track which keys are safe to receive BACK from branches.
    base_state = dict(state)

    if not candidates:
        # ── Fallback: single un-scoped branch (original behaviour) ──────────
        return [
            Send(
                "rca",
                {
                    **base_state,
                    "rca_target_component": "ALL",
                    "rca_branch_id": "ALL",
                    "rca_branch_total_budget": PER_BRANCH_TOTAL_BUDGET,
                    "rca_branch_tool_count": 0,
                    "rca_hyp_per_hypothesis_budget": PER_BRANCH_HYP_BUDGET,
                    "rca_duplicates": 0,
                    "rca_fingerprints": [],
                    "rca_hypothesis_calls": {},
                    "rca_messages": [],   # Always start empty; rca_llm_node builds the full initial prompt
                    "rca_completed": False,
                    # Unconditional override — must replace whatever **base_state
                    # contributed, never merely supplement it (see resolver docstring).
                    "critic_feedback_for_rca": _resolve_branch_feedback(critic_feedback, "ALL"),
                },
            )
        ]

    # ── Fan-out: one branch per candidate component ──────────────────────────
    return [
        Send(
            "rca",
            {
                **base_state,
                "rca_target_component": component,
                "rca_branch_id": component,
                "rca_branch_total_budget": PER_BRANCH_TOTAL_BUDGET,
                "rca_branch_tool_count": 0,
                "rca_hyp_per_hypothesis_budget": PER_BRANCH_HYP_BUDGET,
                # Reset branch-local counters — each branch starts clean
                "rca_duplicates": 0,
                "rca_fingerprints": [],
                "rca_hypothesis_calls": {},
                "rca_messages": [],   # Always start empty; rca_llm_node builds the full initial prompt
                "rca_completed": False,
                # Unconditional override — must replace whatever **base_state
                # contributed, never merely supplement it (see resolver docstring).
                "critic_feedback_for_rca": _resolve_branch_feedback(critic_feedback, component),
            },
        )
        for component in candidates
    ]

# Unnecessary Code Deprecated - Kept for future extensibility
def _build_initial_messages(
    component: str | None,
    critic_feedback: str | None,
) -> list:
    """Deprecated: critic feedback is now injected directly in rca_llm_node.

    Kept as a no-op for backwards compatibility in case of future extension.
    Always returns an empty list — rca_llm_node detects the empty state as
    'first turn' and builds the full initial prompt from scratch, including
    any critic_feedback_for_rca present in state.
    """
    return []

