import asyncio
import json
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from devops_agent.core.llm_provider import LLMFactory
from devops_agent.orchestrator.state import InvestigationState
from ..agents.critic_agent import CriticAgent
from ..agents.schemas import CriticVerdict

class SubmitCriticVerdicts(BaseModel):
    """Submit the verdicts for the mismatched evidence items."""
    verdicts: List[CriticVerdict] = Field(default_factory=list)

async def critic_agent_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    agent = CriticAgent()
    
    mismatched_ids = state.get("mismatched_items", [])
    
    # We only want to review evidence items that are in mismatched_ids
    # But wait, where do we get the actual evidence items from?
    # We need to look in state["final_report"]? No, the evidence items were submitted in the last node.
    # Actually, the deterministic_scoring_node receives SubmitEvidenceReport from rca_agent_node, but the state
    # itself doesn't store the EvidenceItems unless we save them. 
    # Let's check state.py. It has `evidence_matrix`. Wait, we removed updated_hypothesis_scores, but we kept evidence_matrix?
    # No, we didn't remove evidence_matrix from state.py, but the LLM now outputs `evidence_items` in SubmitEvidenceReport.
    # We should retrieve them from state.
    
    # We will assume that `deterministic_scoring_node` saves the `evidence_items` to `state["evidence_matrix"]` or similar.
    # Let's say `deterministic_scoring_node` saves them in `state["current_evidence_items"]` for the critic to review.
    # Let's fetch them from state.
    
    mismatched_items = state.get("current_evidence_items_to_review", [])
    
    bundle = agent.construct_prompt_bundle(state, mismatched_items)
    
    llm = LLMFactory.get_llm("critic").bind_tools([SubmitCriticVerdicts])
    
    critic_system_prompt = """ROLE
--------
You are the Critic Agent. Your job is to review evidence classifications made by the RCA agent that conflict with the raw mathematical data.

CONSTRAINTS & RULES
--------
1. You MUST output a verdict for each provided mismatched evidence item.
2. If the raw data does not match the directional_support claimed, you must reject it with a critique using the 'reject_with_critique' verdict.
3. Your 'rationale' will be sent directly to the RCA Agent. Explain clearly why its reasoning was flawed based on the numbers.
4. You must call SubmitCriticVerdicts to finish."""
    
    messages = [
        SystemMessage(content=critic_system_prompt),
        HumanMessage(content=f"Review the following mismatched evidence classifications:\n{json.dumps(bundle, default=str)}")
    ]
    
    while True:
        response = None
        for attempt in range(3):
            try:
                response = await llm.ainvoke(messages, config=config)
                break
            except Exception as e:
                if attempt == 2: raise e
                await asyncio.sleep(1 * (2 ** attempt))
        
        if not response.tool_calls:
            messages.append(HumanMessage(content="You did not call any tools. You must call SubmitCriticVerdicts to finish."))
            continue
            
        for tc in response.tool_calls:
            if tc["name"] == "SubmitCriticVerdicts":
                args = tc["args"]
                verdicts = args.get("verdicts", [])
                
                # Make sure we convert them to dicts for the state
                if verdicts and not isinstance(verdicts[0], dict):
                    verdicts = [v.model_dump() if hasattr(v, "model_dump") else v.dict() for v in verdicts]
                    
                # Generate feedback message for RCA agent
                critique_lines = []
                for v in verdicts:
                    if v.get("verdict") == "reject_with_critique":
                        critique_lines.append(f"Evidence ID {v.get('evidence_item_id')}: {v.get('rationale')}")
                        
                updates = {
                    "critic_verdicts": verdicts,
                    "current_node": "critic",
                    "rca_completed": False,
                }

                if critique_lines:
                    # Write to a top-level field so the feedback survives the next
                    # rca fan-out dispatch. rca_messages is branch-local and gets
                    # reset on each Send — writing directly to it here would be lost.
                    feedback = (
                        "CRITIC REJECTION: Your evidence report contained mathematical "
                        "contradictions. Please re-evaluate the following items based on "
                        "this feedback and submit a new report:\n"
                        + "\n".join(critique_lines)
                    )
                    updates["critic_feedback_for_rca"] = feedback

                return updates
        
        # If we reach here, no valid tool call was found
        return {"current_node": "critic", "rca_completed": False}
