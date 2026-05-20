"""
tests/test_evals.py

LLM-as-a-judge evaluation suite for the AIRS agent.
Proves that the agent correctly investigates the mock incident and synthesises
an accurate root cause before hitting the HITL interrupt.
"""

import os
import uuid
import json
from pathlib import Path

import pytest
import httpx
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agent.state import make_initial_state
from agent.json_parser import parse_json_robust


class EvaluationResult(BaseModel):
    is_correct: bool = Field(
        description="True if the root cause meets the expected criteria, False otherwise."
    )
    reasoning: str = Field(
        description="A short explanation of why the RCA passed or failed."
    )


async def evaluate_rca(root_cause: str, expected_criteria: str) -> EvaluationResult:
    """Uses an LLM to evaluate the accuracy of the agent's Root Cause Analysis."""
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        max_retries=5
    ).bind(response_format={"type": "json_object"})
    
    prompt = f"""\
You are an expert Site Reliability Engineer evaluating an autonomous agent's incident response.

Evaluate the agent's synthesized Root Cause Analysis (RCA) string below using the provided Grading Note.
Do not demand exact literal substring matches unless explicitly required by the Grading Note.
Evaluate based on semantic correctness, factual accuracy, and diagnostic logic.

Grading Note:
{expected_criteria}

Agent's RCA:
---
{root_cause}
---

Respond ONLY with a JSON object matching this schema:
{{
  "is_correct": boolean,
  "reasoning": "Detailed explanation of why the RCA passed or failed based on the Grading Note"
}}
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    data = parse_json_robust(response.content)
    return EvaluationResult.model_validate(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures():
    """Load the shared JSON fixtures."""
    fixtures_path = Path(__file__).parent.parent / "mock_enterprise" / "fixtures.json"
    with fixtures_path.open() as f:
        return json.load(f)

@pytest.mark.asyncio
@pytest.mark.parametrize("incident_key, expected_criteria", [
    (
        "db_connection_exhaustion",
        "Primary Criteria: The agent must identify that the database connection pool was exhausted or timed out.\nKey Solution Ingredients: Look for acknowledgement of a connection pool timeout and a long-running transaction or connection leak as the root cause.\nAcceptable Ambiguity: The agent does not need to quote the exact tx ID verbatim, as long as the semantic concept of a connection leak blocking the pool is captured."
    ),
    (
        "kubernetes_oom_killed",
        "Primary Criteria: The agent must identify both the application-level failure and the infrastructure-level enforcement action.\nKey Solution Ingredients: Look for acknowledgement of a memory leak or heap exhaustion, coupled with explicit recognition of container eviction by the orchestration layer.\nAcceptable Ambiguity: While the presence of 'Exit code 137' is highly preferred for perfect scores, an answer that accurately states the pod was 'terminated by the kernel due to OOM constraints' is semantically equivalent and should receive a passing grade."
    ),
    (
        "dns_resolution_failure",
        "Primary Criteria: The agent must successfully identify that the downstream connection failure was caused by an upstream configuration or throttling issue.\nKey Solution Ingredients: Look for explicit acknowledgement of DNS throttling, rate limiting, or CoreDNS timeouts as the causal factor.\nAcceptable Ambiguity: The agent does not need to quote the exact log line verbatim, provided it explicitly names CoreDNS and the concept of rate limiting as the primary catalyst for the subsequent socket.gaierror."
    ),
])
async def test_agent_root_cause_accuracy(graph, fixtures, incident_key, expected_criteria):
    """
    End-to-end evaluation test.
    Injects the mock alert, runs the graph until it pauses for HITL approval,
    extracts the drafted plan, and uses an LLM judge to verify the RCA accuracy.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Set the active incident on the mock API
    base_url = os.getenv("MOCK_API_BASE_URL", "http://127.0.0.1:8001")
    async with httpx.AsyncClient() as client:
        await client.post(f"{base_url}/active_incident", json={"incident_key": incident_key})

    # Load specific alert payload
    alert_payload = fixtures["incidents"][incident_key]["alert"]
    initial_state = make_initial_state(alert_payload)
    
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
    eval_result = await evaluate_rca(remediation_plan.root_cause, expected_criteria)
    
    print(f"[Test] LLM Judge Result: is_correct={eval_result.is_correct}")
    print(f"[Test] LLM Judge Reasoning: {eval_result.reasoning}")
    
    # Assert correctness
    assert eval_result.is_correct is True, (
        f"LLM Judge failed the RCA.\n"
        f"Reasoning: {eval_result.reasoning}\n"
        f"RCA: {remediation_plan.root_cause}"
    )
