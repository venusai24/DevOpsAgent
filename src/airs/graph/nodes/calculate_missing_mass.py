"""
Missing Mass Calculator — Module 1.9.

Computes the epistemic gap (missing mass) at each investigation hop.
This is the Information Pursuit driver: high missing mass = more investigation needed.

Missing mass formula (Phase 1 proxy):
    M_t(λ) = 1 - Σ(evidence_contribution_k) / target_coverage

Evidence contribution:
    Each unique (entity_type, signal_type) pair covered reduces missing mass.
    Causal chain nodes contribute more (weighted by 2x).

Missing mass reduction per hop:
    Δ_t = M_{t-1} - M_t

Plateau detection:
    If Δ_t < ε for 3 consecutive hops → plateau → trigger QUERY_PLAYBOOK.
"""
from __future__ import annotations

import logging
from typing import Optional

from airs.models.investigation import InvestigationState
from airs.models.pursuit import InformationPursuitState

log = logging.getLogger(__name__)

# Target evidence categories for a complete investigation
# Each category is (entity_type, signal_type) pair representing a unique signal
_TARGET_CATEGORIES = [
    "metrics_error_rate",
    "metrics_latency",
    "metrics_saturation",
    "logs_application_error",
    "traces_service_call_graph",
    "k8s_state_pod_events",
    "metrics_dependency_health",
    "logs_upstream_downstream",
]

_TOTAL_TARGET_CATEGORIES = len(_TARGET_CATEGORIES)


def compute_missing_mass(
    state: InvestigationState,
    epsilon: float = 0.05,
) -> InformationPursuitState:
    """
    Compute updated InformationPursuitState for the current hop.

    Args:
        state:   Current investigation state.
        epsilon: Plateau detection threshold.

    Returns:
        Updated InformationPursuitState.
    """
    prev_pursuit = state.pursuit_state
    active_nodes = state.insight_tiers.active
    causal_chain_ids = set(state.graph.causal_chain_ids)

    # Compute covered evidence categories from active nodes
    covered: set[str] = set()
    for node in active_nodes:
        entity = node.entity.entity_type.value.lower()
        signal = node.signal_source.value.lower()

        # Map signal + entity to coverage categories
        cat = _map_to_category(signal, entity, node.context)
        if cat:
            # Causal chain nodes cover their category and adjacent ones
            if node.node_id in causal_chain_ids:
                covered.add(cat)
                # Adjacent categories if causal
                if "metrics" in cat:
                    covered.add("metrics_dependency_health")
            else:
                covered.add(cat)

    # Coverage fraction
    coverage_fraction = min(1.0, len(covered) / _TOTAL_TARGET_CATEGORIES)
    new_missing_mass = 1.0 - coverage_fraction

    # Compute delta
    prev_mass = prev_pursuit.current_missing_mass
    delta = abs(prev_mass - new_missing_mass)

    # Plateau detection
    consecutive_low = prev_pursuit.consecutive_low_delta
    if delta < epsilon and state.total_hop_count > 0:
        consecutive_low += 1
    else:
        consecutive_low = 0  # Reset on meaningful progress

    filled = list(covered)
    open_cats = [c for c in _TARGET_CATEGORIES if c not in covered]

    log.debug(
        "Missing mass: %.3f → %.3f (Δ=%.3f, consecutive_low=%d)",
        prev_mass, new_missing_mass, delta, consecutive_low,
    )

    return InformationPursuitState(
        current_missing_mass=new_missing_mass,
        previous_missing_mass=prev_mass,
        missing_mass_delta=delta,
        epsilon_threshold=epsilon,
        consecutive_low_delta=consecutive_low,
        filled_evidence_categories=filled,
        open_evidence_categories=open_cats,
    )


def get_priority_evidence_gap(pursuit: InformationPursuitState) -> Optional[str]:
    """
    Return the highest-priority unfilled evidence category.
    Priority order follows the investigation protocol:
    metrics first → logs → traces → K8s state.
    """
    priority_order = [
        "metrics_error_rate",
        "metrics_latency",
        "logs_application_error",
        "traces_service_call_graph",
        "k8s_state_pod_events",
        "metrics_saturation",
        "metrics_dependency_health",
        "logs_upstream_downstream",
    ]
    for cat in priority_order:
        if cat in pursuit.open_evidence_categories:
            return cat
    return None


def _map_to_category(
    signal: str, entity: str, context: dict
) -> Optional[str]:
    """Map signal/entity/context to a coverage category name."""
    # Metrics mappings
    if signal == "metrics":
        content = context
        metric = str(content.get("metric_name", "")).lower()
        if any(k in metric for k in ["error", "fail", "4xx", "5xx"]):
            return "metrics_error_rate"
        if any(k in metric for k in ["latency", "duration", "p99", "p95"]):
            return "metrics_latency"
        if any(k in metric for k in ["cpu", "memory", "utilization", "saturation"]):
            return "metrics_saturation"
        if any(k in metric for k in ["upstream", "downstream", "dependency"]):
            return "metrics_dependency_health"
        return "metrics_error_rate"  # Default metrics bucket

    # Logs mappings
    if signal == "logs":
        return "logs_application_error"

    # Traces mappings
    if signal == "traces":
        return "traces_service_call_graph"

    # K8s state mappings
    if signal == "k8s_state":
        return "k8s_state_pod_events"

    return None
