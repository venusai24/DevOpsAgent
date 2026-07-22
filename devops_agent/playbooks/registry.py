import os
from glob import glob

import yaml

from devops_agent.orchestrator.agents.schemas import Playbook

PLAYBOOK_REGISTRY: dict[str, Playbook] = {}

def load_playbooks(playbooks_dir: str = None) -> dict[str, Playbook]:
    global PLAYBOOK_REGISTRY
    if not playbooks_dir:
        # Default to the same directory as this file
        playbooks_dir = os.path.dirname(os.path.abspath(__file__))
        
    yaml_files = glob(os.path.join(playbooks_dir, "*.yaml"))
    for file_path in yaml_files:
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)
                if data:
                    playbook = Playbook(**data)
                    PLAYBOOK_REGISTRY[playbook.scenario_id] = playbook
        except Exception as e:
            print(f"Error loading playbook {file_path}: {e}")
            
    return PLAYBOOK_REGISTRY

# Initial load on import
load_playbooks()
