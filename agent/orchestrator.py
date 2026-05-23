"""
agent/orchestrator.py

Wires all AIRS LangGraph nodes into the Hybrid Memory Architecture graph.

New 14-Node Architecture:
  ┌──────────────┐
  │     START    │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ triage_node  │ → classify severity (P0-P3)
  └──────┬───────┘
         │ conditional: P0 → escalate, else → perception
         ▼
  ┌────────────────┐
  │ perception_node│ → L1/L2/L3 log classification + NeSy routing decision
  └──────┬─────────┘
         ▼
  ┌───────────────────┐
  │ topology_agent    │ → EKG dependency traversal (no LLM)
  └──────┬────────────┘
         ▼
  ┌───────────────────┐
  │ diagnostic_agent  │ → CBR retrieval: find similar historical cases
  └──────┬────────────┘
         │ conditional: CBR_GUIDED/SYMBOLIC → skip investigate,
         │              NEURAL_FULL → investigate loop
         ▼
  ┌───────────────────┐◄──────────────────────┐
  │ investigate_node  │                        │  (NEURAL_FULL only)
  └──────┬────────────┘                        │
         │ conditional: INVESTIGATION_COMPLETE → extract, else loop ──────┘
         ▼
  ┌──────────────┐
  │ extract_node │ → verbatim evidence extraction
  └──────┬───────┘
         ▼
  ┌──────────────────┐
  │  logic_agent     │ → symbolic hypothesis pruning (no LLM)
  └──────┬───────────┘
         ▼
  ┌────────────────────┐
  │ remediation_agent  │ → CBR-adapted OR LLM-generated plan
  └──────┬─────────────┘
         ▼
  ┌────────────┐
  │ risk_agent │ → blast radius estimation + execution strategy
  └──────┬─────┘
         ▼
  ┌──────────────────┐
  │ policy_check     │ → Terraform HCL dry-run + invariant check
  └──────┬───────────┘
         │ conditional: blocked → reject | require_approval → approval | canary → canary | direct → approval
         ▼
  ┌───────────────┐
  │ approval_node │ → interrupt() HITL pause
  └──────┬────────┘
         │ conditional: approved → canary/direct, rejected → reject
         ▼
  ┌────────────────────┐         ┌────────────────────┐
  │ canary_execute     │   OR    │ direct_execute      │
  └──────┬─────────────┘         └──────┬─────────────┘
         ▼                               ▼
  ┌────────────┐
  │ retain     │ → store resolved case in CBR (continuous learning)
  └──────┬─────┘
         ▼
       END
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
try:
    from langgraph.checkpoint.postgres.aio import PostgresSaver
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    PostgresSaver = None
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from config import settings

from agent.nodes import (
    MAX_RETRIES,
    # Triage
    triage_node,
    escalate_node,
    # Perception
    perception_node,
    # Reasoning multi-agents
    topology_agent_node,
    diagnostic_agent_node,
    logic_agent_node,
    remediation_agent_node,
    # Risk & Policy
    risk_agent_node,
    policy_check_node,
    # Execution
    canary_execute_node,
    direct_execute_node,
    retain_node,
    # HITL & Reject
    reject_node,
    approval_node,
    # NEURAL_FULL fallback nodes
    investigate_node,
    should_continue_investigation,
    extract_node,
)
from agent.state import GraphState, RemediationPlan
from agent.reasoning.nesym_router import ReasoningPathway, NeuroSymbolicRouter

logger = logging.getLogger(__name__)

_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", settings.CHECKPOINT_DB_PATH)
_NESYM_ROUTER = NeuroSymbolicRouter()


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_triage(state: dict) -> str:
    """P0 → escalate first, else → perception."""
    severity: str = state.get("severity", "")
    if severity == "P0":
        logger.info("[router] P0 → escalate_node")
        return "escalate"
    logger.info("[router] severity=%s → perception_node", severity)
    return "perception"


def _route_after_diagnostic(state: dict) -> str:
    """
    NeSy routing decision after CBR retrieval.

    SYMBOLIC_FAST / CBR_GUIDED → skip investigation, go straight to logic_agent.
    NEURAL_FULL → full ReAct investigation loop.
    """
    perception_stats = state.get("perception_stats", {})
    cbr_confidence = state.get("cbr_confidence", 0.0)
    primary_template = state.get("primary_log_template", "")

    routing = _NESYM_ROUTER.route(perception_stats, cbr_confidence, primary_template)
    logger.info(
        "[router] NeSy routing decision: %s (cbr=%.0f%% l1=%.0f%%)",
        routing.pathway.value, cbr_confidence * 100, routing.l1_rate * 100,
    )

    if routing.pathway in (ReasoningPathway.SYMBOLIC_FAST, ReasoningPathway.CBR_GUIDED):
        return "logic"       # Skip investigation → symbolic pruning
    return "investigate"     # Full ReAct loop


def _route_after_plan(state: dict) -> str:
    """High-risk plans (no rollback) → reject, safe → risk_agent."""
    plan: RemediationPlan | None = state.get("remediation_plan")
    if plan and plan.is_high_risk:
        logger.warning("[router] High-risk plan (no rollback) → reject_node")
        return "reject"
    return "risk"


def _route_after_policy(state: dict) -> str:
    """
    Route based on execution_strategy set by risk_agent + policy_check:
      blocked          → reject
      require_approval → approval
      canary           → approval (operator must still confirm)
      direct           → approval (same)
    """
    strategy = state.get("execution_strategy", "require_approval")
    logger.info("[router] execution_strategy=%s", strategy)
    if strategy == "blocked":
        return "reject"
    return "approval"   # All non-blocked strategies go through HITL


def _route_after_approval(state: dict) -> str:
    """After HITL: approved → canary or direct based on strategy, rejected → reject."""
    if not state.get("is_approved", False):
        return "reject"
    strategy = state.get("execution_strategy", "require_approval")
    if strategy == "canary":
        return "canary"
    return "direct"


# ---------------------------------------------------------------------------
# Legacy postmortem helper (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def _build_postmortem(
    alert: dict,
    plan: RemediationPlan | None,
    service: str,
    severity: str,
    executed_steps: list[str],
) -> str:
    """Build a structured markdown postmortem document."""
    triggered_at = alert.get("triggered_at", "unknown")
    alert_title = alert.get("title", "Unknown incident")
    alert_id = alert.get("id", "N/A")

    root_cause = plan.root_cause if plan else "Root cause analysis unavailable."
    mttr = plan.estimated_mttr_minutes if plan else None
    postmortem_summary = (plan.postmortem_summary if plan else "") or root_cause[:300]
    rollback_cmd = plan.rollback_command if plan else "N/A"

    steps_md = "\n".join(executed_steps) if executed_steps else "No steps executed."

    return f"""# Incident Postmortem — {alert_id}

**Service**: {service}
**Severity**: {severity}
**Alert**: {alert_title}
**Incident opened**: {triggered_at}
**MTTR**: {f'{mttr} minutes' if mttr else 'N/A'}
**Status**: Resolved ✅

---

## Summary

{postmortem_summary}

---

## Root Cause Analysis

{root_cause}

---

## Remediation Steps Executed

{steps_md}

---

## Rollback Command (available if regression occurs)

```bash
{rollback_cmd}
```

---

## Action Items

- [ ] Add integration test covering this failure scenario
- [ ] Review on-call runbook for `{service}`
- [ ] Set alert threshold 20% below failure threshold
- [ ] Schedule blameless post-incident review with on-call team

---

*Postmortem generated automatically by AIRS (Autonomous Incident Response System).*
*Human review and sign-off required before closing this incident.*
"""


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def build_graph() -> AsyncIterator:
    """
    Async context manager: constructs and yields the compiled AIRS LangGraph.

    Automatically selects:
      - PostgreSQL checkpointer when DATABASE_URL is set (production).
      - SQLite checkpointer otherwise (local dev / demo mode).

    Usage:
        async with build_graph() as graph:
            config = {"configurable": {"thread_id": "incident-001"}}
            async for event in graph.astream_events(initial_state, config, version="v2"):
                ...

    Crash-safe resume:
        async with build_graph() as graph:
            await graph.ainvoke(
                Command(resume={"approved": True}),
                config,  # same thread_id
            )
    """
    if settings.DATABASE_URL:
        if not HAS_POSTGRES:
            raise ImportError(
                "DATABASE_URL is configured but 'langgraph-checkpoint-postgres' is not installed. "
                "Run: pip install -U 'langgraph-checkpoint-postgres[psycopg]'"
            )
        async with PostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
            await checkpointer.setup()
            graph = _build_compiled_graph(checkpointer)
            logger.info(
                "[orchestrator] Graph compiled with %d nodes (Postgres checkpointer).",
                len(graph.nodes),
            )
            yield graph
    else:
        async with AsyncSqliteSaver.from_conn_string(_DB_PATH) as checkpointer:
            graph = _build_compiled_graph(checkpointer)
            logger.info(
                "[orchestrator] Graph compiled with %d nodes (SQLite checkpointer: %s).",
                len(graph.nodes), _DB_PATH,
            )
            yield graph


def _build_compiled_graph(checkpointer):
    """
    Pure graph construction — separated from the context manager so tests
    can inject an in-memory checkpointer without spawning DB connections.

    Node topology (14 nodes + START/END):

      START → triage → [escalate →] perception → topology_agent → diagnostic_agent
        → [investigate ↺ |] extract → logic_agent → remediation_agent → risk_agent
        → policy_check → [reject | approval → [reject | canary | direct] → retain] → END
    """
    graph = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # Register all nodes
    # ------------------------------------------------------------------

    # Phase 0: Triage
    graph.add_node("triage", triage_node)
    graph.add_node("escalate", escalate_node)

    # Phase 3: Perception
    graph.add_node("perception", perception_node)

    # Phase 1: EKG Topology
    graph.add_node("topology_agent", topology_agent_node)

    # Phase 2: CBR Diagnostic
    graph.add_node("diagnostic_agent", diagnostic_agent_node)

    # Phase 4: NEURAL_FULL investigate loop (used only when CBR confidence is low)
    graph.add_node("investigate", investigate_node)
    graph.add_node("extract", extract_node)

    # Phase 4: Symbolic pruning + plan generation
    graph.add_node("logic_agent", logic_agent_node)
    graph.add_node("remediation_agent", remediation_agent_node)

    # Phase 5a: Risk assessment
    graph.add_node("risk_agent", risk_agent_node)

    # Phase 5b: Policy-as-Code
    graph.add_node("policy_check", policy_check_node)

    # HITL approval gate
    graph.add_node("approval", approval_node)

    # Execution
    graph.add_node("canary", canary_execute_node)
    graph.add_node("direct_execute", direct_execute_node)

    # Continuous learning
    graph.add_node("retain", retain_node)

    # Rejection / escalation terminal
    graph.add_node("reject", reject_node)

    # ------------------------------------------------------------------
    # Register edges
    # ------------------------------------------------------------------

    # Entry: START → triage
    graph.add_edge(START, "triage")

    # Triage → (escalate for P0 | perception for all others)
    graph.add_conditional_edges(
        "triage",
        _route_after_triage,
        {"escalate": "escalate", "perception": "perception"},
    )

    # Escalate always falls through to perception (after Slack notification)
    graph.add_edge("escalate", "perception")

    # Perception → topology (always sequential)
    graph.add_edge("perception", "topology_agent")

    # Topology → diagnostic (always sequential)
    graph.add_edge("topology_agent", "diagnostic_agent")

    # Diagnostic → (logic [SYMBOLIC/CBR] | investigate [NEURAL_FULL])
    graph.add_conditional_edges(
        "diagnostic_agent",
        _route_after_diagnostic,
        {"logic": "logic_agent", "investigate": "investigate"},
    )

    # Investigation ReAct loop (NEURAL_FULL only)
    graph.add_conditional_edges(
        "investigate",
        should_continue_investigation,
        {"investigate": "investigate", "extract": "extract"},
    )

    # Extract → logic (join both pathways at logic_agent)
    graph.add_edge("extract", "logic_agent")

    # Logic → remediation (always sequential)
    graph.add_edge("logic_agent", "remediation_agent")

    # Remediation → risk assessment
    graph.add_conditional_edges(
        "remediation_agent",
        _route_after_plan,
        {"risk": "risk_agent", "reject": "reject"},
    )

    # Risk → policy check
    graph.add_edge("risk_agent", "policy_check")

    # Policy check → (reject [blocked] | approval [all other strategies])
    graph.add_conditional_edges(
        "policy_check",
        _route_after_policy,
        {"reject": "reject", "approval": "approval"},
    )

    # Approval → (reject [operator rejected] | canary | direct_execute)
    graph.add_conditional_edges(
        "approval",
        _route_after_approval,
        {
            "reject": "reject",
            "canary": "canary",
            "direct": "direct_execute",
        },
    )

    # Both execution paths converge at retain (continuous learning)
    graph.add_edge("canary", "retain")
    graph.add_edge("direct_execute", "retain")
    graph.add_edge("retain", END)
    graph.add_edge("reject", END)

    # ------------------------------------------------------------------
    # Compile with checkpointer
    # ------------------------------------------------------------------
    return graph.compile(checkpointer=checkpointer)
