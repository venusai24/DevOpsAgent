"""EvidenceAgent for Stages 5-6."""

from typing import Any

from ..state import InvestigationState


class EvidenceAgent:
    """Executes Stages 5-6 (Evidence Collection and Graph Refinement)."""
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        
    def construct_prompt_bundle(self, state: InvestigationState) -> dict[str, Any]:
        """Extracts bounded context for Stages 5-6."""
        return {
            "role": "EVIDENCE_AGENT",
            "inputs": {
                "time_range": state.get("time_range"),
                "baseline_registry_ref": state.get("baseline_registry_ref", ""),
                "component_registry": state.get("component_registry", {}),
                "declared_topology_graph": state.get("declared_topology_graph", {}),
                "blast_radius": state.get("blast_radius"),
                "investigation_cluster": state.get("investigation_cluster", []),
                "ranked_hypotheses": state.get("ranked_hypotheses", []),
                "T0": state.get("T0"),
                "symptom_pattern": state.get("symptom_pattern")
            }
        }
        
    def parse_output(self, llm_response: dict[str, Any]) -> dict[str, Any]:
        """Validates and extracts only Stages 5-6 state fields."""
        return {
            "evidence_matrix": llm_response.get("evidence_matrix", {}),
            "updated_hypothesis_scores": llm_response.get("updated_hypothesis_scores", {}),
            "eliminated_hypotheses": llm_response.get("eliminated_hypotheses", []),
            "surviving_hypotheses": llm_response.get("surviving_hypotheses", []),
            "primary_bottleneck": llm_response.get("primary_bottleneck"),
            "refined_dependency_graph": llm_response.get("refined_dependency_graph", {}),
            "undeclared_dependencies": llm_response.get("undeclared_dependencies", []),
            "propagation_verified_pairs": llm_response.get("propagation_verified_pairs", []),
            "investigation_state": llm_response.get("investigation_state", "active"),
            "current_node": "evidence_collection",
            "investigation_gaps": llm_response.get("investigation_gaps", [])
        }
