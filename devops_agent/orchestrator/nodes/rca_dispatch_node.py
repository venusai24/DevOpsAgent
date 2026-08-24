"""RCA Fan-Out Dispatcher.

Uses LangGraph's Send API to spawn one independent RCA branch per affected component
identified by the Triage Agent. Each branch receives its own isolated budget counters
and message history so branches never interfere with each other.

If triage produces no candidates (e.g. loop-prevention fallback), a single un-scoped
branch is dispatched — preserving the original sequential behaviour.
"""

from __future__ import annotations

from langgraph.types import Send

from ..state import InvestigationState

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


def dispatch_rca_fan_out(state: InvestigationState) -> list[Send]:
    """Return one Send per affected component, each with isolated branch state.

    LangGraph will execute all returned Sends concurrently. When every branch
    finishes, control passes to the ``merge_rca`` node, which consolidates results
    into the shared InvestigationState.
    """
    candidates: list[str] = state.get("affected_component_candidates", [])

    # Build critic feedback prefix if the critic has flagged issues from a
    # previous RCA pass (i.e. the graph looped through critic → rca again).
    critic_feedback: str | None = state.get("critic_feedback_for_rca")

    # Clear per_branch_rca_results so the merge node starts fresh on re-dispatch.
    # We must explicitly zero this out; reducers only append, never reset.
    # Build the base state that is passed INTO each branch.
    # We pass the full state so branches can read everything they need.
    # However, we track which keys are safe to receive BACK from branches.
    base_state = dict(state)
    base_state["per_branch_rca_results"] = []

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
            },
        )
        for component in candidates
    ]


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

