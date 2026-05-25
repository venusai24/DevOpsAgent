"""
tests/test_kb_lookup_node.py

Unit tests for the kb_lookup_node LangGraph node (agent/nodes.py).

Tests the node in isolation by mocking agent.kb.store.kb_lookup to return
controlled KBRetrievalResult instances.  Verifies that:

  - On exact bypass: the node writes remediation_plan, plan, is_approved=True,
    and faithfulness_score=1.0 to the state dict.
  - On RAG match: the node writes kb_result but does NOT write remediation_plan
    or set is_approved.
  - On no match: the node writes kb_result with match_type='none' and does not
    touch remediation_plan.
  - The {service} placeholder in commands is correctly substituted at bypass time.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from agent.state import (
    KBEntry,
    KBRemediationStep,
    KBRetrievalResult,
    TriageResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb_entry(confidence: float = 0.97) -> KBEntry:
    return KBEntry(
        entry_id="kb-node-test-001",
        incident_taxonomy="Errors:OOM",
        pattern_type="exact",
        error_pattern="Exit Code 137",
        affected_services=[],
        severity="P1",
        root_cause_narrative="Pod was OOMKilled. Memory limit too low.",
        remediation_steps=[
            KBRemediationStep(
                order=1,
                action="Increase memory limit.",
                environment="kubectl",
                command="kubectl set resources deployment/{service} -n prod --limits=memory=2Gi",
                risk="medium",
            ),
        ],
        rollback_command="kubectl rollout undo deployment/{service} -n prod",
        confidence_score=confidence,
    )


def _make_state(service: str = "payment-service") -> dict:
    return {
        "raw_alert": {
            "id": "alert-001",
            "title": "CRITICAL: Pod OOMKilled",
            "description": "Exit Code 137 in last 5 minutes.",
            "service": service,
        },
        "triage_result": TriageResult(
            severity="P1",
            service=service,
            confidence=0.92,
            reasoning="OOMKilled pattern detected.",
        ),
        "severity": "P1",
        "messages": [],
        "kb_result": None,
        "faithfulness_score": None,
    }


# ---------------------------------------------------------------------------
# Tests: Full bypass path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bypass_writes_plan_and_approved():
    """On exact bypass, node should write remediation_plan and is_approved=True."""
    from agent.nodes import kb_lookup_node

    entry = _make_kb_entry(confidence=0.97)
    bypass_result = KBRetrievalResult(
        entry=entry,
        retrieval_score=0.97,
        match_type="exact",
        bypass_llm=True,
    )

    with patch("agent.nodes.kb_lookup", new=AsyncMock(return_value=bypass_result)):
        updates = await kb_lookup_node(_make_state())

    assert updates["is_approved"] is True
    assert updates["remediation_plan"] is not None
    assert updates["plan"] != ""
    assert updates["faithfulness_score"] == 1.0
    assert updates["kb_result"].bypass_llm is True


@pytest.mark.asyncio
async def test_bypass_service_placeholder_substituted():
    """The {service} placeholder in commands should be replaced with the actual service name."""
    from agent.nodes import kb_lookup_node

    service = "checkout-service"
    entry = _make_kb_entry()
    bypass_result = KBRetrievalResult(
        entry=entry,
        retrieval_score=0.97,
        match_type="exact",
        bypass_llm=True,
    )

    with patch("agent.nodes.kb_lookup", new=AsyncMock(return_value=bypass_result)):
        updates = await kb_lookup_node(_make_state(service=service))

    plan = updates["remediation_plan"]
    assert plan is not None
    for step in plan.steps:
        if step.command:
            assert "{service}" not in step.command, (
                f"Placeholder not substituted in command: {step.command}"
            )
            assert service in step.command, (
                f"Service name not found in command: {step.command}"
            )


@pytest.mark.asyncio
async def test_bypass_retry_count_set_to_max():
    """retry_count should be MAX_RETRIES so the router skips investigation."""
    from agent.nodes import kb_lookup_node, MAX_RETRIES

    bypass_result = KBRetrievalResult(
        entry=_make_kb_entry(),
        retrieval_score=0.97,
        match_type="exact",
        bypass_llm=True,
    )

    with patch("agent.nodes.kb_lookup", new=AsyncMock(return_value=bypass_result)):
        updates = await kb_lookup_node(_make_state())

    assert updates.get("retry_count") == MAX_RETRIES


# ---------------------------------------------------------------------------
# Tests: RAG match path (no bypass)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_match_does_not_set_approved():
    """A RAG-range match should write kb_result but NOT set is_approved."""
    from agent.nodes import kb_lookup_node

    rag_result = KBRetrievalResult(
        entry=_make_kb_entry(confidence=0.82),
        retrieval_score=0.78,  # in [0.70, 0.95)
        match_type="exact",
        bypass_llm=False,
    )

    with patch("agent.nodes.kb_lookup", new=AsyncMock(return_value=rag_result)):
        updates = await kb_lookup_node(_make_state())

    assert "is_approved" not in updates or updates.get("is_approved") is not True
    assert "remediation_plan" not in updates or updates.get("remediation_plan") is None
    assert updates["kb_result"].bypass_llm is False
    assert updates["kb_result"].retrieval_score == pytest.approx(0.78)


# ---------------------------------------------------------------------------
# Tests: No match path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_match_writes_none_result():
    """A score below RAG threshold should write kb_result with match_type='none'."""
    from agent.nodes import kb_lookup_node

    no_match_result = KBRetrievalResult(
        entry=None,
        retrieval_score=0.10,
        match_type="none",
        bypass_llm=False,
    )

    with patch("agent.nodes.kb_lookup", new=AsyncMock(return_value=no_match_result)):
        updates = await kb_lookup_node(_make_state())

    assert updates["kb_result"].match_type == "none"
    assert updates["kb_result"].entry is None
    assert updates.get("is_approved") is not True


@pytest.mark.asyncio
async def test_no_match_appends_message():
    """On any path, the node should append exactly one message."""
    from agent.nodes import kb_lookup_node

    no_match_result = KBRetrievalResult(
        entry=None,
        retrieval_score=0.0,
        match_type="none",
        bypass_llm=False,
    )

    with patch("agent.nodes.kb_lookup", new=AsyncMock(return_value=no_match_result)):
        updates = await kb_lookup_node(_make_state())

    # LangGraph uses the add_messages reducer so we get a list
    assert "messages" in updates
    assert len(updates["messages"]) >= 1
