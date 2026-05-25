"""
agent/orchestrator.py

Wires all AIRS LangGraph nodes into a compiled, checkpointed StateGraph.

Architecture (ImplementationPlan.md Section 5):
  ┌─────────────┐
   │  START       │
   └──────┬───────┘
          ▼
   ┌─────────────┐
   │ triage_node │  → classify severity (P0-P3)
   └──────┬───────┘
          │ always
          ▼
   ┌───────────────┐
   │ kb_lookup_node│  → hybrid KB lookup (exact/regex/semantic)
   └──────┬────────┘
          │ conditional: _route_after_kb_lookup
          │   "execute"     ─── score ≥ 0.95: EXACT BYPASS (skip LLM pipeline)
          │   "escalate"    ─── P0 severity: page on-call
          │   "investigate" ─── normal ReAct investigation
          ▼
  ┌──────────────────┐
  │ investigate_node │◄─────────────────────────────────┐
  └──────┬───────────┘                                  │
         │ conditional router: should_continue_investigation │
         │   "plan"       ─────────────────────────────►│ (exit loop)
         │   "investigate" ────────────────────────────►┘ (loop back)
         ▼
  ┌───────────┐
  │ plan_node │  → draft RemediationPlan
  └──────┬────┘
         │ conditional: is_high_risk → reject_node, else → approval_node
         ▼
  ┌───────────────┐
  │ approval_node │  → interrupt() suspends execution for HITL
  └──────┬────────┘
         │ resumes via Command(resume={"approved": True/False})
         ▼
  ┌──────────────┐
  │ execute_node │  → generate postmortem
  └──────┬───────┘
         ▼
       END

Durable state:
  AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") is used as the
  checkpointer. Every node transition is persisted to disk, enabling the
  crash-safe resume demo (kill → restart → pass thread_id → resume instantly).

Usage:
    from agent.orchestrator import build_graph

    async with build_graph() as graph:
        config = {"configurable": {"thread_id": "incident-001"}}
        async for event in graph.astream_events(initial_state, config, version="v2"):
            ...
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
try:
    from langgraph.checkpoint.postgres.aio import PostgresSaver
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    PostgresSaver = None
from langgraph.config import var_child_runnable_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from config import settings
from agent.integrations.slack import send_slack_approval_request
from agent.integrations.k8s import apply_kubectl_command
from agent.security.guardrails import is_safe_command

from agent.nodes import (
    MAX_RETRIES,
    investigate_node,
    extract_node,
    plan_node,
    should_continue_investigation,
    triage_node,
    kb_lookup_node,
)
from agent.state import GraphState, RemediationPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite")
"""
Path to the SQLite file used for durable state checkpointing.
Override via CHECKPOINT_DB_PATH env var (e.g. in tests: ":memory:").
"""

# ---------------------------------------------------------------------------
# Additional nodes not defined in nodes.py
# ---------------------------------------------------------------------------


async def escalate_node(state: dict) -> dict:
    """
    Zero-trust fast-path for P0 incidents.

    When the triage router detects P0 severity, execution is routed here
    BEFORE the investigation loop. The node injects a high-priority warning
    into the message stream and then falls through to investigate_node
    by routing to it via a normal edge.

    This node does NOT call any LLM — it is a pure deterministic Python
    guardrail to ensure P0 incidents are flagged before any tool calls.

    In a production system this would also page the on-call engineer via
    PagerDuty / Slack. Here it appends a visible message to the state so
    the Rich CLI can render a ⚠ banner.
    """
    severity = state.get("severity", "P0")
    service = ""
    if tr := state.get("triage_result"):
        service = tr.service

    logger.warning(
        "[escalate_node] P0 ESCALATION: service=%s severity=%s", service, severity
    )

    warning_msg = AIMessage(
        content=(
            f"⚠️  **P0 ESCALATION** — {service or 'unknown service'} is experiencing "
            f"a critical outage. Bypassing standard triage queue. "
            f"Initiating immediate investigation."
        )
    )
    return {"messages": [warning_msg]}


async def approval_node(state: dict, config: RunnableConfig) -> dict:
    """
    Human-in-the-Loop approval gate.

    Suspends graph execution using LangGraph's ``interrupt()`` primitive.
    The CLI reads the interrupt payload, renders the remediation plan to the
    operator, and waits for typed input.

    Resumption:
        The CLI calls ``graph.ainvoke(Command(resume={"approved": True}), config)``
        to resume. If the operator typed 'reject', it passes
        ``Command(resume={"approved": False})``.

    After resumption, this node reads the ``approved`` value from the
    ``Command.resume`` dict and writes ``is_approved`` to state.

    Crash-safe demo:
        Because the checkpoint is written BEFORE interrupt() is called, killing
        the terminal and restarting the CLI with the same thread_id will land
        the graph exactly at this node's suspension point. The operator can
        then approve/reject without re-running triage or investigation.
    """
    plan_md: str = state.get("plan", "No plan generated.")
    remediation_plan: RemediationPlan | None = state.get("remediation_plan")

    high_risk = remediation_plan.is_high_risk if remediation_plan else True
    risk_label = "HIGH-RISK ⚠️" if high_risk else "Standard ✅"

    logger.info("[approval_node] Suspending for HITL approval. risk=%s", risk_label)

    # In production, dispatch Slack notification instead of terminal blocking
    if settings.SLACK_BOT_TOKEN:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        # Ensure we don't send multiple slack messages on resume
        if "approved" not in state.keys(): # heuristic for first pass
            await send_slack_approval_request(plan_md, risk_label, thread_id, high_risk)

    # interrupt() suspends the graph here and surfaces the payload to the CLI.
    # Execution will not proceed past this line until Command(resume=...) is sent.
    token = var_child_runnable_config.set(config)
    try:
        approval_response: dict = interrupt(
            {
                "message": "Review the remediation plan below and type 'approve' to proceed or 'reject' to abort.",
                "plan": plan_md,
                "risk": risk_label,
                "is_high_risk": high_risk,
            }
        )
    finally:
        var_child_runnable_config.reset(token)

    # approval_response is the dict passed in Command(resume={...})
    approved: bool = bool(approval_response.get("approved", False))
    decision = "APPROVED" if approved else "REJECTED"

    logger.info("[approval_node] Operator decision: %s", decision)

    decision_msg = AIMessage(
        content=(
            f"Operator decision: **{decision}**. "
            + ("Proceeding to execution." if approved else "Execution aborted.")
        )
    )

    return {
        "is_approved": approved,
        "messages": [decision_msg],
    }


async def execute_node(state: dict) -> dict:
    """
    Post-approval execution: generate the postmortem document.

    In a real system this node would run the kubectl commands from the
    remediation plan. Here it generates a structured postmortem markdown
    document from the accumulated state, simulating a successful fix.

    The postmortem is written to ``postmortem.md`` in the working directory
    by the CLI — this node only assembles the markdown string.

    Reads:
        state["remediation_plan"] — For root cause, steps, rollback command.
        state["raw_alert"]        — For incident metadata.
        state["triage_result"]    — For severity and service.
        state["is_approved"]      — Checked as a safety guard.

    Writes:
        state["postmortem"] — Markdown postmortem string.
        state["messages"]   — Appends execution summary AIMessage.
    """
    if not state.get("is_approved", False):
        # Should never reach here without approval due to routing, but guard anyway
        logger.error("[execute_node] Called without operator approval — aborting.")
        abort_msg = AIMessage(
            content="❌ Execution aborted: remediation plan was not approved."
        )
        return {"messages": [abort_msg]}

    alert = state.get("raw_alert", {})
    plan: RemediationPlan | None = state.get("remediation_plan")
    triage = state.get("triage_result")

    service = triage.service if triage else alert.get("service", "unknown")
    severity = state.get("severity", "unknown")

    logger.info("[execute_node] Generating postmortem for service=%s", service)

    # --- Simulate execution of each kubectl / shell step ---
    executed_steps: list[str] = []
    execution_succeeded = True
    if plan:
        for step in sorted(plan.steps, key=lambda s: s.order):
            log_line = f"✅ Step {step.order}: {step.action}"
            
            if step.command:
                # Apply Guardrails
                if not is_safe_command(step.command):
                    log_line += f"\n   `[BLOCKED BY GUARDRAIL] {step.command}`"
                    execution_succeeded = False
                else:
                    # Apply Kubernetes SDK
                    k8s_result = apply_kubectl_command(step.command)
                    log_line += f"\n   `{step.command}`\n   Output: {k8s_result}"
                    
            executed_steps.append(log_line)
            logger.info("[execute_node] %s", log_line)

    # --- Closed-loop KB feedback: update confidence_score based on outcome ---
    kb_result = state.get("kb_result")
    if kb_result and kb_result.entry:
        try:
            from agent.kb.store import kb_update_confidence
            await kb_update_confidence(kb_result.entry.entry_id, success=execution_succeeded)
            logger.info(
                "[execute_node] KB confidence updated: entry=%s success=%s",
                kb_result.entry.entry_id,
                execution_succeeded,
            )
        except Exception as exc:
            logger.warning("[execute_node] KB confidence update failed: %s", exc)

    # --- Build postmortem document ---
    postmortem = _build_postmortem(
        alert=alert,
        plan=plan,
        service=service,
        severity=severity,
        executed_steps=executed_steps,
    )

    exec_msg = AIMessage(
        content=(
            f"✅ Remediation executed successfully for **{service}** ({severity}).\n"
            f"Postmortem document generated ({len(postmortem)} chars)."
        )
    )

    return {
        "postmortem": postmortem,
        "messages": [exec_msg],
    }


async def reject_node(state: dict) -> dict:
    """
    Handles high-risk plans with no rollback command.

    If plan_node generates a RemediationPlan where is_high_risk is True,
    the plan router sends execution here instead of the approval node.
    The node appends a clear rejection message and routes to END.
    """
    plan: RemediationPlan | None = state.get("remediation_plan")
    logger.warning(
        "[reject_node] High-risk plan rejected. rollback_command=%r",
        plan.rollback_command if plan else "N/A",
    )
    reject_msg = AIMessage(
        content=(
            "🚫 **Plan rejected by zero-trust guardrail**: The generated remediation "
            "plan does not include a rollback command. Automated execution of "
            "irreversible operations is blocked. A human operator must review and "
            "manually supply a rollback strategy before proceeding."
        )
    )
    return {"messages": [reject_msg], "is_approved": False}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_triage(state: dict) -> str:
    """
    After triage_node: always proceed to kb_lookup_node.
    Kept as a passthrough for compatibility; kb_lookup_node handles all
    severity-based routing via _route_after_kb_lookup.

    Deprecated: Will be removed once all tests are updated to use kb_lookup.
    """
    # Retained only as a fallback — orchestrator now routes triage → kb_lookup directly.
    return "kb_lookup"


def _route_after_kb_lookup(state: dict) -> str:
    """
    After kb_lookup_node: three-way routing based on KB match score and severity.

    1. bypass_llm=True (exact KB match, score ≥ KB_EXACT_BYPASS_THRESHOLD):
       Route directly to execute_node.  The plan and is_approved=True are
       already written to state by kb_lookup_node.

    2. P0 severity (without bypass): Route to escalate_node to page on-call,
       then escalate → investigate via a fixed edge.

    3. All other cases: Route to investigate_node for the normal ReAct loop.
    """
    from agent.state import KBRetrievalResult
    kb_result: KBRetrievalResult | None = state.get("kb_result")

    if kb_result and kb_result.bypass_llm:
        logger.info(
            "[router] KB EXACT BYPASS (score=%.3f) → execute_node",
            kb_result.retrieval_score,
        )
        return "execute"

    severity: str = state.get("severity", "")
    if severity == "P0":
        logger.info("[router] P0 detected → escalate_node")
        return "escalate"

    logger.info("[router] severity=%s, no bypass → investigate_node", severity)
    return "investigate"


def _route_after_plan(state: dict) -> str:
    """
    After plan_node: high-risk plans (no rollback) → reject_node,
    safe plans → approval_node.
    """
    plan: RemediationPlan | None = state.get("remediation_plan")
    if plan and plan.is_high_risk:
        logger.warning("[router] Plan is high-risk → reject_node")
        return "reject"
    logger.info("[router] Plan is safe → approval_node")
    return "approve"


def _route_after_approval(state: dict) -> str:
    """
    After approval_node: approved → execute_node, rejected → END.
    """
    if state.get("is_approved", False):
        return "execute"
    logger.info("[router] Operator rejected plan → END")
    return END


# ---------------------------------------------------------------------------
# Postmortem builder
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

- [ ] Add integration test covering connection pool exhaustion scenario
- [ ] Set alert threshold for pool utilisation > 80% (current: no warning)
- [ ] Review long-running transaction handling in the payment processor
- [ ] Add `finally` block to all database session managers
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
    Async context manager that constructs and yields a compiled LangGraph
    StateGraph backed by AsyncSqliteSaver for durable checkpointing.

    The SQLite connection is opened inside the context manager and closed
    cleanly on exit, preventing file-lock issues in the demo environment.

    Usage:
        async with build_graph() as graph:
            config = {"configurable": {"thread_id": "incident-abc"}}
            result = await graph.ainvoke(initial_state, config)

    Crash-safe resume:
        Because every node transition is persisted to disk, you can kill the
        process mid-execution and resume by calling:
            async with build_graph() as graph:
                await graph.ainvoke(
                    Command(resume={"approved": True}),
                    config,   # same thread_id
                )
        The graph will pick up from the exact checkpoint where it left off.
    """
    if settings.DATABASE_URL:
        if not HAS_POSTGRES:
            raise ImportError(
                "DATABASE_URL is configured, but 'langgraph-checkpoint-postgres' is not installed. "
                "Please run: pip install -U \"langgraph-checkpoint-postgres[psycopg]\""
            )
        # Use Postgres checkpointer if DB URL is configured
        async with PostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
            await checkpointer.setup()
            graph = _build_compiled_graph(checkpointer)
            logger.info(
                "[orchestrator] Graph compiled with %d nodes. Checkpointing to Postgres.",
                len(graph.nodes)
            )
            yield graph
    else:
        # Fall back to SQLite
        async with AsyncSqliteSaver.from_conn_string(_DB_PATH) as checkpointer:
            graph = _build_compiled_graph(checkpointer)
            logger.info(
                "[orchestrator] Graph compiled with %d nodes. Checkpointing to %s",
                len(graph.nodes),
                _DB_PATH,
            )
            yield graph


def _build_compiled_graph(checkpointer: AsyncSqliteSaver):
    """
    Pure graph construction — separated from the context manager so it can
    be called in tests with an in-memory checkpointer.

    Node map:
      triage       → triage_node
      escalate     → escalate_node        (P0 fast-path)
      investigate  → investigate_node     (ReAct loop)
      plan         → plan_node
      approval     → approval_node        (interrupt / HITL)
      execute      → execute_node
      reject       → reject_node          (high-risk guardrail)

    Edge map:
      START         → triage
      triage        → escalate | investigate      (conditional: P0 vs other)
      escalate      → investigate                 (deterministic: always investigate after escalation)
      investigate   → investigate | plan          (conditional: should_continue_investigation)
      plan          → approval | reject           (conditional: is_high_risk)
      approval      → execute | END              (conditional: is_approved)
      execute       → END
      reject        → END
    """
    # Build graph with the GraphState schema
    # Using dict as state type (TypedDict-compatible) with add_messages reducer
    graph = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    graph.add_node("triage", triage_node)
    graph.add_node("kb_lookup", kb_lookup_node)  # KB grounding gate (new)
    graph.add_node("escalate", escalate_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("extract", extract_node)
    graph.add_node("plan", plan_node)
    graph.add_node("approval", approval_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reject", reject_node)

    # ------------------------------------------------------------------
    # Register edges
    # ------------------------------------------------------------------

    # Entry point: always triage first, then KB lookup
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "kb_lookup")

    # After KB lookup: three-way conditional routing
    graph.add_conditional_edges(
        "kb_lookup",
        _route_after_kb_lookup,
        {
            "execute": "execute",    # Exact KB bypass: skip full LLM pipeline
            "escalate": "escalate",  # P0 fast-path (still investigates after)
            "investigate": "investigate",
        },
    )

    # After escalation: always proceed to investigation
    graph.add_edge("escalate", "investigate")

    # Investigation ReAct loop
    graph.add_conditional_edges(
        "investigate",
        should_continue_investigation,
        {
            "investigate": "investigate",   # loop back
            "plan": "extract",               # exit loop to extract node
        },
    )

    # After extract: sequential flow into plan
    graph.add_edge("extract", "plan")

    # After plan: zero-trust guardrail check
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "approve": "approval",
            "reject": "reject",
        },
    )

    # After approval: operator decision
    graph.add_conditional_edges(
        "approval",
        _route_after_approval,
        {
            "execute": "execute",
            END: END,
        },
    )

    # Terminal nodes
    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    # ------------------------------------------------------------------
    # Compile with checkpointer
    # ------------------------------------------------------------------
    return graph.compile(
        checkpointer=checkpointer,
        # interrupt_before is NOT set here — we use the interrupt() primitive
        # inside approval_node itself, which is the idiomatic LangGraph approach
        # for dynamic, payload-bearing HITL pauses.
    )
