"""RCAAgent for Stages 5-8."""

from typing import Any

from ..state import InvestigationState


class RCAAgent:
    """Executes Stages 5-8 (Evidence Collection, Graph Refinement, Root Cause, Reporting)."""
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        
    def construct_prompt_bundle(self, state: InvestigationState) -> dict[str, Any]:
        """Extracts bounded context for Stages 5-8."""
        return {
            "role": "RCA_AGENT",
            "inputs": {
                "time_range": state.get("time_range"),
                "baseline_registry_ref": state.get("baseline_registry_ref", ""),
                "component_registry": state.get("component_registry", {}),
                "declared_topology_graph": state.get("declared_topology_graph", {}),
                "stack_kpi_map": state.get("stack_kpi_map", {}),

                "blast_radius": state.get("blast_radius"),
                "investigation_cluster": state.get("investigation_cluster", []),
                "ranked_hypotheses": state.get("ranked_hypotheses", []),
                "T0": state.get("T0"),
                "symptom_pattern": state.get("symptom_pattern")
            }
        }
        
    def parse_output(self, llm_response: dict[str, Any]) -> dict[str, Any]:
        """Validates and extracts only Stages 5-8 state fields."""
        return {
            "evidence_matrix": llm_response.get("evidence_matrix", {}),
            "updated_hypothesis_scores": llm_response.get("updated_hypothesis_scores", {}),
            "eliminated_hypotheses": llm_response.get("eliminated_hypotheses", []),
            "surviving_hypotheses": llm_response.get("surviving_hypotheses", []),
            "primary_bottleneck": llm_response.get("primary_bottleneck"),
            "refined_dependency_graph": llm_response.get("refined_dependency_graph", {}),
            "undeclared_dependencies": llm_response.get("undeclared_dependencies", []),
            "propagation_verified_pairs": llm_response.get("propagation_verified_pairs", []),
            "root_cause_candidate": llm_response.get("root_cause_candidate"),
            "causal_chain": llm_response.get("causal_chain", []),
            "confidence_level": llm_response.get("confidence_level"),
            "unconfirmed_links": llm_response.get("unconfirmed_links", []),
            "final_report": llm_response.get("final_report"),
            "investigation_state": llm_response.get("investigation_state", "active"),
            "current_node": "rca",
            "investigation_gaps": llm_response.get("investigation_gaps", [])
        }
