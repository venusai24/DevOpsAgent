import json
from typing import Any

from devops_agent.orchestrator.state import InvestigationState

class CriticAgent:
    def construct_prompt_bundle(self, state: InvestigationState, mismatched_items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "role": "CRITIC_AGENT",
            "inputs": {
                "mismatched_evidence_items": mismatched_items
            }
        }
