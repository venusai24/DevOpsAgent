"""TriageAgent for Stages 1-4."""

from typing import Any

from ..state import InvestigationState


class TriageAgent:
    """Executes Stages 1-4 (Triage and Hypothesis Generation)."""
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        
    def _compress_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # 1. Compress Component Registry
        reg = inputs.get("component_registry", {})
        if reg:
            components = list(reg.values())
            common_attrs = dict(components[0])
            for comp in components[1:]:
                for k in list(common_attrs.keys()):
                    if k not in comp or comp[k] != common_attrs[k]:
                        del common_attrs[k]
            if common_attrs:
                compressed_reg = {}
                for name, data in reg.items():
                    compressed_reg[name] = {k: v for k, v in data.items() if k not in common_attrs}
                inputs["component_registry"] = {
                    "shared_attributes": common_attrs,
                    "components": compressed_reg
                }
                
        # 2. Compress topology graphs (drop nodes with no dependencies)
        for graph_key in ["declared_topology_graph", "discovered_topology_graph"]:
            topo = inputs.get(graph_key, {})
            if topo:
                filtered_topo = {k: v for k, v in topo.items() if v}
                if filtered_topo:
                    inputs[graph_key] = filtered_topo
                else:
                    inputs.pop(graph_key, None)
            else:
                inputs.pop(graph_key, None)
                
        # 2b. If dependencies are unknown, prefer discovered topology over declared
        explicit_symptoms = inputs.get("explicit_symptoms")
        if isinstance(explicit_symptoms, dict):
            deps_unknown = explicit_symptoms.get("dependencies_unknown", False)
        else:
            deps_unknown = getattr(explicit_symptoms, "dependencies_unknown", False)
            
        if deps_unknown and "discovered_topology_graph" in inputs:
            inputs.pop("declared_topology_graph", None)
            
        # 3. Compress tc_to_operation_map (drop identity maps where key == operation)
        tc_map = inputs.get("tc_to_operation_map", {})
        if tc_map:
            filtered_tc_map = {k: v for k, v in tc_map.items() if v.get("operation") != k}
            inputs["tc_to_operation_map"] = filtered_tc_map
            
        return inputs

    def construct_prompt_bundle(self, state: InvestigationState) -> dict[str, Any]:
        """Extracts bounded context for Stages 1-4 per ADR-002 section 7.3."""
        inputs = {
            "time_range": state.get("time_range"),
            "explicit_symptoms": state.get("explicit_symptoms", {}),
            "component_registry": state.get("component_registry", {}),
            "tc_to_operation_map": state.get("tc_to_operation_map", {}),
            "stack_kpi_map": state.get("stack_kpi_map", []),
            "baseline_registry_ref": state.get("baseline_registry_ref", ""),
            "declared_topology_graph": state.get("declared_topology_graph", {}),
            "discovered_topology_graph": state.get("discovered_topology_graph", {})
        }
        return {
            "role": "TRIAGE_AGENT",
            "inputs": self._compress_inputs(inputs)
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
            "investigation_state": llm_response.get("investigation_state", "active"),
            "current_node": "triage"
        }
