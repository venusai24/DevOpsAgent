"""
tests/test_evals.py

LLM-as-a-judge evaluation suite for the AIRS agent.
Proves that the agent correctly investigates the mock incident and synthesises
an accurate root cause before hitting the HITL interrupt.
"""

import os
import uuid

import pytest
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agent.state import make_initial_state
from cli.main import _DEMO_ALERT


class EvaluationResult(BaseModel):
    is_correct: bool = Field(
        description="True if the root cause accurately identifies database connection pool exhaustion/timeouts and cites a connection leak or long-running transaction."
    )
    reasoning: str = Field(
        description="A short explanation of why the RCA passed or failed."
    )


async def evaluate_rca(root_cause: str) -> EvaluationResult:
    """Uses an LLM to evaluate the accuracy of the agent's Root Cause Analysis."""
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        max_retries=5
    ).with_structured_output(EvaluationResult)
    
    prompt = f"""\
You are an expert Site Reliability Engineer evaluating an autonomous agent's incident response.

The agent was presented with a P0 database connection pool exhaustion incident.
Evaluate the agent's synthesized Root Cause Analysis (RCA) string below.

Criteria for PASS (is_correct=true):
1. Must identify that the database connection pool was exhausted or timed out.
2. Must cite the root cause as a long-running transaction or connection leak.

Agent's RCA:
---
{root_cause}
---
"""
    return await llm.ainvoke([HumanMessage(content=prompt)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_root_cause_accuracy(graph):
    """
    End-to-end evaluation test.
    Injects the mock alert, runs the graph until it pauses for HITL approval,
    extracts the drafted plan, and uses an LLM judge to verify the RCA accuracy.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = make_initial_state(_DEMO_ALERT)
    
    # Run the graph. It will pause when it hits the interrupt() in approval_node.
    # ainvoke will return the state as it stands at the point of interruption.
    state = await graph.ainvoke(initial_state, config)
    
    # Extract the generated remediation plan
    remediation_plan = state.get("remediation_plan")
    
    # Check that a plan was actually generated before the pause
    assert remediation_plan is not None, "Agent failed to generate a remediation plan."
    assert remediation_plan.root_cause, "Agent's remediation plan is missing a root cause."
    
    print(f"\n[Test] Agent RCA: {remediation_plan.root_cause}")
    
    # Evaluate the RCA using the LLM judge
    eval_result = await evaluate_rca(remediation_plan.root_cause)
    
    print(f"[Test] LLM Judge Result: is_correct={eval_result.is_correct}")
    print(f"[Test] LLM Judge Reasoning: {eval_result.reasoning}")
    
    # Assert correctness
    assert eval_result.is_correct is True, (
        f"LLM Judge failed the RCA.\n"
        f"Reasoning: {eval_result.reasoning}\n"
        f"RCA: {remediation_plan.root_cause}"
    )
