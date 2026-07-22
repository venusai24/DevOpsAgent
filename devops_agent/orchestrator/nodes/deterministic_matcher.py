from typing import Any

from devops_agent.playbooks.registry import PLAYBOOK_REGISTRY

from ..agents.schemas import MatchResult
from ..state import InvestigationState


def match_playbook(playbook, kpi_map: dict[str, Any]) -> MatchResult:
    """
    Generic deterministic matcher that checks if the playbook's kpi_categories
    are significantly present in the anomalous kpi_map.
    """
    total_components = len(kpi_map)
    matched_components = []
    
    for cid, data in kpi_map.items():
        categories = data.get("available_categories", {})
        # If any of the playbook's required categories are in this component's anomalies
        if any(cat in categories for cat in playbook.kpi_categories):
            matched_components.append(cid)
            
    # Simple signal strength: ratio of affected components that match the signature
    signal_strength = len(matched_components) / total_components if total_components > 0 else 0.0
    
    # Heuristic thresholds based on tier
    threshold = 0.3 if playbook.difficulty_tier == "easy" else 0.1
    
    # Hardcoded precedence (e.g. jvm_oom vs high_cpu)
    if playbook.scenario_id == "high_cpu":
        # If JVM anomalies are also very strong, discount CPU strength (assume GC thrashing)
        jvm_strength = sum(1 for cid, data in kpi_map.items() if "jvm" in data.get("available_categories", {})) / total_components if total_components > 0 else 0
        if jvm_strength > 0.5:
            signal_strength *= 0.5
            
    fired = signal_strength >= threshold
    
    return MatchResult(
        scenario_id=playbook.scenario_id,
        fired=fired,
        signal_strength=min(1.0, signal_strength),
        matched_components=matched_components,
        trigger_signals={"matched_ratio": signal_strength}
    )

def deterministic_matcher_node(state: InvestigationState) -> dict[str, Any]:
    kpi_map = state.get("discovered_kpi_map", {})
    results = []
    
    for playbook in PLAYBOOK_REGISTRY.values():
        result = match_playbook(playbook, kpi_map)
        results.append(result.model_dump())
        
    return {"match_results": results}
