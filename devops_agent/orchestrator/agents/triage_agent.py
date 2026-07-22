"""TriageAgent for Stages 1-4."""

from typing import Any

from ..state import InvestigationState


class TriageAgent:
    """Executes Stages 1-4 (Triage and Hypothesis Generation)."""
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        
    def construct_prompt_bundle(self, state: InvestigationState) -> dict[str, Any]:
        """Extracts bounded context for Stages 1-4 per ADR-002 section 7.3."""
        return {
            "role": "TRIAGE_AGENT",
            "inputs": {
                "time_range": state.get("time_range"),
                "explicit_symptoms": state.get("explicit_symptoms", {}),
                "component_registry": state.get("component_registry", {}),
                "healthy_components": state.get("healthy_components", ()),
                "stack_kpi_map": state.get("stack_kpi_map", []),
                "baseline_registry_ref": state.get("baseline_registry_ref", ""),
                "topology_graph": state.get("discovered_topology_graph") or state.get("declared_topology_graph", {})
            }
        }
        
    def parse_output(self, llm_response: dict[str, Any]) -> dict[str, Any]:
        """Validates and extracts only Stages 1-4 state fields."""
        return {
            "blast_radius": llm_response.get("blast_radius", "localized"),
            "blast_radius_qualifier": llm_response.get("blast_radius_qualifier", "simultaneous"),
            "symptom_pattern": llm_response.get("symptom_pattern", "unknown"),
            "anomalous_tc_values": llm_response.get("anomalous_tc_values", {}),
            "affected_component_candidates": llm_response.get("affected_component_candidates", []),
            "T0": llm_response.get("T0"),
            "T0_sources": llm_response.get("T0_sources", {}),
            "leading_indicators": llm_response.get("leading_indicators", {}),
            "incident_state": llm_response.get("incident_state", "ongoing"),
            "investigation_cluster": llm_response.get("investigation_cluster", []),
            "concurrent_incident_clusters": llm_response.get("concurrent_incident_clusters", []),
            "boundary_ambiguous_components": llm_response.get("boundary_ambiguous_components", []),
            "dependency_graph": llm_response.get("dependency_graph", {}),
            "ranked_hypotheses": llm_response.get("ranked_hypotheses", []),
            "rejected_playbooks": llm_response.get("rejected_playbooks", []),
            "investigation_state": llm_response.get("investigation_state", "active"),
            "current_node": "triage"
        }
