"""ReasoningAgent for Stages 7-8."""

from typing import Any

from ..state import InvestigationState


class ReasoningAgent:
    """Executes Stages 7-8 (Causal Reasoning and Reporting)."""
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        
    def construct_prompt_bundle(self, state: InvestigationState) -> dict[str, Any]:
        """Extracts bounded context for Stages 7-8."""
        return {
            "role": "REASONING_AGENT",
            "inputs": {
                "time_range": state.get("time_range"),
                "component_registry": state.get("component_registry", {}),
                "investigation_cluster": state.get("investigation_cluster", []),
                "T0": state.get("T0"),
                "evidence_matrix": state.get("evidence_matrix", {}),
                "surviving_hypotheses": state.get("surviving_hypotheses", []),
                "primary_bottleneck": state.get("primary_bottleneck"),
                "refined_dependency_graph": state.get("refined_dependency_graph", {})
            }
        }
        
    def parse_output(self, llm_response: dict[str, Any]) -> dict[str, Any]:
        """Validates and extracts only Stages 7-8 state fields."""
        return {
            "root_cause_candidate": llm_response.get("root_cause_candidate"),
            "causal_chain": llm_response.get("causal_chain", []),
            "confidence_level": llm_response.get("confidence_level"),
            "unconfirmed_links": llm_response.get("unconfirmed_links", []),
            "final_report": llm_response.get("final_report"),
            "investigation_state": llm_response.get("investigation_state", "active"),
            "current_node": "causal_reasoning"
        }
