"""ContextAssemblerService for Stage 0."""

import re
from typing import Any

from ..state import InvestigationState


class ContextAssemblerService:
    """Executes Stage 0 to build the initial context artifacts deterministically."""
    
    def __init__(self):
        pass
        
    def determine_role(self, name: str) -> dict[str, str]:
        name_lower = name.lower()
        if re.search(r'tomcat', name_lower):
            return {"role": "app_server", "stack_type": "jvm", "layer": "app"}
        elif re.search(r'mysql', name_lower):
            return {"role": "database", "stack_type": "mysql", "layer": "data"}
        elif re.search(r'redis', name_lower):
            return {"role": "cache", "stack_type": "redis", "layer": "data"}
        elif re.search(r'apache|ig|mg', name_lower):
            return {"role": "gateway", "stack_type": "apache", "layer": "network"}
        elif re.search(r'docker', name_lower):
            return {"role": "container", "stack_type": "docker", "layer": "infra"}
        else:
            return {"role": "unknown", "stack_type": "unknown", "layer": "unknown"}

    def assemble(self, state: InvestigationState, discovered_cmdb_ids: list[str], discovered_tc_values: list[str]) -> dict[str, Any]:
        """Validates and extracts only Stage 0 state fields using deterministic logic."""
        
        component_registry = {}
        for cid in discovered_cmdb_ids + discovered_tc_values:
            if cid not in component_registry:
                role_info = self.determine_role(cid)
                component_registry[cid] = {
                    "name": cid,
                    "role": role_info["role"],
                    "layer": role_info["layer"],
                    "stack_type": role_info["stack_type"],
                    "role_source": "regex_mapping",
                    "cmdb_declared_role": "unknown"
                }
                
        stack_kpi_map = [
            {"stack_type": "jvm", "kpis": ["cpu", "memory", "tomcat_request", "tomcat_session", "tomcat_thread"]},
            {"stack_type": "mysql", "kpis": ["cpu", "memory", "network", "disk", "process"]},
            {"stack_type": "redis", "kpis": ["cpu", "memory", "redis_memory", "redis_clients", "redis_perf"]},
            {"stack_type": "apache", "kpis": ["cpu", "memory", "network"]},
            {"stack_type": "docker", "kpis": ["cpu", "memory", "network", "disk"]},
        ]
        
        return {
            "component_registry": component_registry,
            "stack_kpi_map": stack_kpi_map,
            "stage_0_gaps": [],
            "current_node": "context_assembly"
        }

