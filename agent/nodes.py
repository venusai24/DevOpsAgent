"""
agent/nodes.py

Pure Python LangGraph node functions for the AIRS graph.

Design rules (ImplementationPlan.md Section 5 & 6):
  - Every node is a plain async function: (state: dict) -> dict.
  - Nodes return ONLY the state keys they modify.
  - All LLM calls use with_structured_output() for type-safe parsing.
  - The investigation node is a ReAct loop capped by retry_count <= MAX_RETRIES.
  - No system prompts for safety; destructive-operation guardrails are in
    deterministic Python code (RemediationPlan.is_high_risk).

Node catalogue:
  triage_node       — Classify severity (P0-P3) and extract the service name.
  investigate_node  — ReAct loop: call get_metrics / get_logs, append telemetry.
  plan_node         — Synthesise telemetry into a typed RemediationPlan.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.state import GraphState, RemediationPlan, RemediationStep, TriageResult
from agent.tools import ALL_TOOLS, get_logs, get_metrics
from agent.json_parser import parse_json_robust
from agent.playbook_cache import (
    lookup_playbook,
    record_successful_playbook,
    substitute_service,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3
"""
Hard cap on investigation loop iterations.
When retry_count reaches this value the router sends the graph directly
to plan_node regardless of whether the LLM has finished calling tools.
Prevents runaway API spend when the LLM is stuck in a tool loop.
"""

_MODEL_NAME: str = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _make_llm(**kwargs: Any) -> ChatGroq:
    """Instantiate a Groq Qwen LLM using the GROQ_API_KEY env var."""
    return ChatGroq(
        model=_MODEL_NAME,
        temperature=0,      # Deterministic outputs for structured extraction
        max_retries=5,      # Resiliency against rate limits
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Node: triage_node
# ---------------------------------------------------------------------------

_TRIAGE_PROMPT = """\
You are a senior Site Reliability Engineer performing initial incident triage.

You will receive a raw PagerDuty-style alert payload. Your ONLY job is to:
1. Assign a severity level (P0, P1, P2, or P3) based on the alert description
   and any available metadata.
2. Identify the canonical name of the affected microservice.
3. Assign a confidence score (0.0–1.0) for your severity classification.
4. Write a single-sentence reasoning that cites specific evidence from the alert.

Severity guidelines:
  P0 — Active customer-facing outage, error rate > 20%, or data loss risk.
  P1 — Major degradation, error rate 5–20%, or SLA breach imminent.
  P2 — Minor degradation, error rate < 5%, no SLA risk.
  P3 — Informational alert, no user impact.

Respond ONLY with a JSON object. Do not add any prose, markdown, or text outside the JSON.
"""

# Friendly JSON template for triage output — avoids triggering Qwen tool-call
# wrapping that occurs when formal JSON schemas are injected into the prompt.
_TRIAGE_JSON_TEMPLATE = """\
{
  "severity": "<P0 | P1 | P2 | P3>",
  "service": "<canonical microservice name from the alert>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence citing specific evidence from the alert>"
}"""


async def triage_node(state: dict) -> dict:
    """
    Classify the incoming alert into a severity level and extract the
    affected service name.

    Reads:
        state["raw_alert"] — PagerDuty-style alert dict.

    Writes:
        state["severity"]      — P-level string, e.g. "P0".
        state["triage_result"] — Full TriageResult Pydantic instance.
        state["messages"]      — Appends HumanMessage (input) + AIMessage.
    """
    raw_alert: dict = state.get("raw_alert", {})
    logger.info(
        "[triage_node] Classifying alert id=%s", raw_alert.get("id", "unknown")
    )

    # Bind JSON mode — do NOT inject a formal JSON schema; Qwen treats
    # formal schemas as function/tool definitions and wraps its response
    # in a tool-call envelope, causing Groq to return 400 tool_use_failed.
    # A plain template string is sufficient and avoids this failure mode.
    llm = _make_llm().bind(response_format={"type": "json_object"})

    human_msg = HumanMessage(
        content=(
            f"{_TRIAGE_PROMPT}\n\n"
            f"## Alert Payload\n```json\n{json.dumps(raw_alert, indent=2)}\n```\n\n"
            f"Respond ONLY with a JSON object matching this exact template:\n"
            f"```\n{_TRIAGE_JSON_TEMPLATE}\n```"
        )
    )

    response = await llm.ainvoke([human_msg])
    data = parse_json_robust(response.content)
    result = TriageResult.model_validate(data)

    # Log low-confidence classifications for observability
    if result.confidence < 0.6:
        logger.warning(
            "[triage_node] Low-confidence classification: severity=%s confidence=%.2f",
            result.severity,
            result.confidence,
        )

    ai_msg = AIMessage(
        content=(
            f"Triage complete. Severity: **{result.severity}** "
            f"| Service: **{result.service}** "
            f"| Confidence: {result.confidence:.0%}\n"
            f"Reasoning: {result.reasoning}"
        )
    )

    logger.info(
        "[triage_node] severity=%s service=%s confidence=%.2f",
        result.severity,
        result.service,
        result.confidence,
    )

    return {
        "severity": result.severity,
        "triage_result": result,
        "messages": [human_msg, ai_msg],
    }


# ---------------------------------------------------------------------------
# Node: investigate_node
# ---------------------------------------------------------------------------

_INVESTIGATE_PROMPT_TEMPLATE = """\
You are an SRE investigator performing root cause analysis for the following
incident:

  Service  : {service}
  Severity : {severity}
  Alert    : {alert_title}

## Investigation Objective
Use the available tools to gather enough telemetry to hand off a complete
picture to the RCA agent. Follow this strategy based strictly on the alert symptoms:

1. Call `get_metrics` with queries relevant ONLY to the symptoms of this specific alert:
   - If the alert mentions database connection pool, DB queries, postgresql, database exhaustion, or query timeouts: query "db_connections".
   - If the alert mentions OOM, OOMKilled, Memory usage, Java heap space, node eviction, or exit code 137: query "memory_utilization" or "cpu_utilization". Do NOT query database connections.
   - If the alert mentions DNS resolution, name resolution failures, CoreDNS, external routing, or upstream API failures: query "dns_latency" or "error_rate". Do NOT query database connections.
   
2. Call `get_logs` with service="{service}" to inspect actual error traces, warning messages, and exceptions from the affected service.

## Telemetry Gathered So Far
{telemetry_so_far}

## Instructions
- Make ONE tool call per response.
- After each tool result is appended, decide whether you need more data.
- When you have sufficient evidence, respond with ONLY the text:
  INVESTIGATION_COMPLETE
  Do NOT add any other text after INVESTIGATION_COMPLETE.
- You have at most {remaining_retries} tool call(s) remaining before the system
  forces you to conclude. Use them wisely.
"""


async def investigate_node(state: dict) -> dict:
    """
    ReAct-style investigation loop: calls get_metrics and get_logs tools,
    accumulates results into state["telemetry"], and increments retry_count.

    The loop runs for a SINGLE tool-call round per invocation. The graph
    router calls this node repeatedly until either:
      a) The LLM signals INVESTIGATION_COMPLETE.
      b) retry_count reaches MAX_RETRIES (hard cap enforced by router).

    This single-step-per-invocation design is idiomatic LangGraph: the
    router, not the node, controls looping. It also means every tool-call
    round is checkpointed individually, enabling crash-safe resumption.

    Playbook Cache:
        On the FIRST invocation of each graph run (retry_count == 0), the node
        checks the playbook cache for a pre-validated tool sequence matching
        the alert's symptom fingerprint. On a HIT:
          - All tool calls are replayed deterministically in a single pass.
          - The LLM is NOT invoked, eliminating hallucination risk and latency.
          - state["retry_count"] is set to MAX_RETRIES to signal the router
            that investigation is complete and transition to plan_node.
        On a MISS, the LLM runs normally. If the investigation reaches
        INVESTIGATION_COMPLETE, the successful tool sequence is written back
        to the cache for future use.

    Reads:
        state["triage_result"] — For service name.
        state["raw_alert"]     — For alert title.
        state["severity"]      — For prompt context.
        state["telemetry"]     — Accumulated telemetry string (may be empty).
        state["retry_count"]   — Current iteration count.
        state["messages"]      — Full message history.

    Writes:
        state["telemetry"]   — Appended with new tool output (if a tool ran).
        state["retry_count"] — Incremented by 1 (or set to MAX_RETRIES on cache hit).
        state["messages"]    — Appended with AI + ToolMessage (if tool ran),
                               or the INVESTIGATION_COMPLETE AIMessage.
    """
    triage: TriageResult | None = state.get("triage_result")
    service: str = triage.service if triage else state.get("raw_alert", {}).get("service", "unknown")
    severity: str = state.get("severity", "P1")
    raw_alert: dict = state.get("raw_alert", {})
    alert_title: str = raw_alert.get("title", "Unknown incident")
    alert_description: str = raw_alert.get("description", "")
    telemetry_so_far: str = state.get("telemetry", "")
    retry_count: int = state.get("retry_count", 0)
    remaining = MAX_RETRIES - retry_count

    logger.info(
        "[investigate_node] iteration=%d/%d service=%s",
        retry_count + 1, MAX_RETRIES, service,
    )

    # ---------------------------------------------------------------
    # Playbook Cache Lookup (only on the FIRST iteration of a run)
    # ---------------------------------------------------------------
    if retry_count == 0:
        cache_result = lookup_playbook(alert_title, alert_description)
        if cache_result is not None:
            fingerprint, raw_sequence = cache_result
            # Substitute the runtime service name into any "{service}" placeholders
            tool_sequence = substitute_service(raw_sequence, service)

            logger.info(
                "[investigate_node] CACHE HIT fingerprint=%r — replaying %d tool calls, skipping LLM",
                fingerprint, len(tool_sequence),
            )

            new_telemetry = telemetry_so_far
            new_messages: list = []
            step_index = 0

            for step in tool_sequence:
                step_index += 1
                tool_name: str = step["tool"]
                tool_args: dict = step["args"]
                # Generate a synthetic tool_call_id so ToolMessage format is valid
                tool_call_id = f"cache_{fingerprint}_{step_index}"

                logger.info(
                    "[investigate_node] Replaying cached step %d: %s(%s)",
                    step_index, tool_name, tool_args,
                )

                try:
                    if tool_name == "get_metrics":
                        tool_result: str = await get_metrics.ainvoke(tool_args)
                    elif tool_name == "get_logs":
                        tool_result = await get_logs.ainvoke(tool_args)
                    else:
                        tool_result = f"[CACHE] Unknown tool in playbook: {tool_name}"
                except Exception as exc:
                    tool_result = f"[CACHE TOOL ERROR] {tool_name}: {exc}"
                    logger.warning(
                        "[investigate_node] Cached tool %s failed: %s — proceeding with partial telemetry",
                        tool_name, exc,
                    )

                # Append as a ToolMessage with a synthetic AI tool-call wrapper
                tool_msg = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
                new_messages.append(tool_msg)

                section_header = (
                    f"\n\n---\n### [CACHED] Step {step_index} — `{tool_name}`"
                    f"({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})\n"
                )
                new_telemetry = new_telemetry + section_header + tool_result

            # Append a synthetic INVESTIGATION_COMPLETE message so the router
            # correctly transitions to extract_node → plan_node.
            complete_msg = AIMessage(
                content="INVESTIGATION_COMPLETE (served from playbook cache)"
            )
            new_messages.append(complete_msg)

            return {
                "telemetry": new_telemetry,
                # Set retry_count to MAX_RETRIES so the router exits the loop
                "retry_count": MAX_RETRIES,
                "messages": new_messages,
            }

    # ---------------------------------------------------------------
    # Cache MISS — run the LLM normally
    # ---------------------------------------------------------------

    # Accumulate the tool calls made during this live run so they can be
    # written back to the cache if the investigation completes successfully.
    live_tool_sequence: list[dict[str, Any]] = state.get("_live_tool_sequence", [])

    prompt = _INVESTIGATE_PROMPT_TEMPLATE.format(
        service=service,
        severity=severity,
        alert_title=alert_title,
        telemetry_so_far=telemetry_so_far if telemetry_so_far else "(none yet)",
        remaining_retries=remaining,
    )

    prior_messages = state.get("messages", [])
    round_messages = [HumanMessage(content=prompt)] + prior_messages

    llm = _make_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    ai_response: AIMessage = await llm_with_tools.ainvoke(round_messages)

    new_messages = [ai_response]
    new_telemetry = telemetry_so_far
    new_retry_count: int = retry_count + 1

    # ---------------------------------------------------------------
    # Case A: LLM wants to call a tool
    # ---------------------------------------------------------------
    if ai_response.tool_calls:
        tool_call = ai_response.tool_calls[0]  # Process one call per round
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(
            "[investigate_node] tool_call name=%s args=%s", tool_name, tool_args
        )

        try:
            if tool_name == "get_metrics":
                tool_result = await get_metrics.ainvoke(tool_args)
            elif tool_name == "get_logs":
                tool_result = await get_logs.ainvoke(tool_args)
            else:
                tool_result = f"Unknown tool: {tool_name}"
        except Exception as exc:
            tool_result = f"[TOOL ERROR] {tool_name}: {exc}"
            logger.warning("[investigate_node] Tool error: %s", exc)

        tool_msg = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        new_messages.append(tool_msg)

        section_header = (
            f"\n\n---\n### Round {new_retry_count} — `{tool_name}`"
            f"({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})\n"
        )
        new_telemetry = new_telemetry + section_header + tool_result

        logger.info(
            "[investigate_node] telemetry length=%d chars", len(new_telemetry)
        )

        # Record this tool call for potential cache write-back later
        live_tool_sequence = live_tool_sequence + [{"tool": tool_name, "args": tool_args}]

    # ---------------------------------------------------------------
    # Case B: LLM signals investigation is complete (no tool call)
    # ---------------------------------------------------------------
    else:
        response_text: str = ai_response.content or ""
        if "INVESTIGATION_COMPLETE" in response_text:
            logger.info("[investigate_node] LLM signalled INVESTIGATION_COMPLETE")
            # Write the validated tool sequence back to the cache for future use
            if live_tool_sequence:
                record_successful_playbook(
                    alert_title=alert_title,
                    alert_description=alert_description,
                    tool_sequence=live_tool_sequence,
                )
        else:
            logger.warning(
                "[investigate_node] No tool call and no INVESTIGATION_COMPLETE. "
                "Response: %s",
                response_text[:200],
            )

    return {
        "telemetry": new_telemetry,
        "retry_count": new_retry_count,
        "messages": new_messages,
        "_live_tool_sequence": live_tool_sequence,
    }


# ---------------------------------------------------------------------------
# Router: should_continue_investigation
# ---------------------------------------------------------------------------


def should_continue_investigation(state: dict) -> str:
    """
    Conditional edge function for the investigation loop.

    Returns "investigate" to loop back, or "plan" to proceed to plan_node.

    Rules (both must be False to proceed to plan):
      1. retry_count < MAX_RETRIES  →  keep investigating.
      2. Last AIMessage contains no tool_calls AND no INVESTIGATION_COMPLETE
         signal  →  still loop (LLM may need another prompt).

    Hard cap: once retry_count >= MAX_RETRIES, always proceed regardless of
    LLM intent. This is the runaway-spend circuit breaker.
    """
    retry_count: int = state.get("retry_count", 0)

    # Hard cap: always exit if at or over limit
    if retry_count >= MAX_RETRIES:
        logger.info(
            "[router] retry_count=%d >= MAX_RETRIES=%d → routing to plan_node",
            retry_count, MAX_RETRIES,
        )
        return "plan"

    # Check last AI message for a tool call or completion signal
    messages = state.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )

    if last_ai is None:
        return "investigate"

    # If the LLM made a tool call in the last round, loop back
    if last_ai.tool_calls:
        return "investigate"

    # If the LLM signalled completion, proceed
    if "INVESTIGATION_COMPLETE" in (last_ai.content or ""):
        return "plan"

    # Fallback: continue investigating
    return "investigate"


# ---------------------------------------------------------------------------
# Node: extract_node
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT_TEMPLATE = """\
You are an expert systems diagnostician. Your ONLY job is to extract exact verbatim strings from the provided Telemetry Evidence. Do NOT invent, paraphrase, or summarize anything.

## Telemetry Evidence
{telemetry}

## Task
1. Sequential Scan: Read the telemetry chronologically.
2. Metric Isolation: Extract the exact values of any anomalous metrics (e.g. "99%", "100 active connections", "312s").
3. Error Isolation: Extract the exact error strings, exception names, and exit codes (e.g. "java.lang.OutOfMemoryError", "Exit code 137", "CoreDNS rate limiting").

Respond ONLY with an XML block containing your raw extracted findings. Use this exact format:
<extracted_evidence>
[Your verbatim quotes here]
</extracted_evidence>
"""

async def extract_node(state: dict) -> dict:
    """
    Step 1 of the Extract-Then-Generate pipeline.
    Reads the raw telemetry string and explicitly isolates key metrics and errors
    into an XML scratchpad. This mitigates attention hijacking and ensures verbatim
    retention of critical identifiers (like exit codes or WARN logs).
    """
    telemetry: str = state.get("telemetry", "No telemetry gathered.")
    logger.info("[extract_node] Extracting key evidence from telemetry")

    llm = _make_llm()  # Standard text output, no JSON mode needed
    
    prompt = _EXTRACT_PROMPT_TEMPLATE.format(telemetry=telemetry)
    human_msg = HumanMessage(content=prompt)

    response = await llm.ainvoke([human_msg])
    
    extracted_text = response.content
    # Simple extraction of the XML block if present
    if "<extracted_evidence>" in extracted_text:
        extracted_text = extracted_text.split("<extracted_evidence>")[1].split("</extracted_evidence>")[0].strip()

    ai_msg = AIMessage(content=f"Extracted evidence:\n{extracted_text}")

    # We store the extracted text back into the state for the plan_node to use
    return {
        "extracted_evidence": extracted_text,
        "messages": [human_msg, ai_msg],
    }

# ---------------------------------------------------------------------------
# Node: plan_node
# ---------------------------------------------------------------------------

_PLAN_PROMPT_TEMPLATE = """\
You are a senior SRE architect drafting an incident remediation plan.

## Incident Context
- **Service**: {service}
- **Severity**: {severity}
- **Alert**: {alert_title}

## Extracted Verbatim Evidence
{extracted_evidence}

## Task
Produce a precise, actionable remediation plan as a JSON object based on the Extracted Verbatim Evidence above.

1. `root_cause` — A single paragraph citing the exact metrics, log messages, and exit codes from the Extracted Verbatim Evidence. Identify the underlying root cause. You MUST construct this paragraph using the exact phrases provided in the evidence block above.
2. `steps` — Ordered list of atomic remediation actions. Each step must include:
   - `order`: integer starting at 1
   - `action`: human-readable description
   - `command`: exact kubectl / shell command (use null if not applicable)
   - `risk`: one of "low", "medium", "high"

3. `rollback_command` — A single shell/kubectl command that fully reverts all
   changes. MUST NOT be empty.

4. `estimated_mttr_minutes` — Your best estimate of time-to-recovery in minutes.

5. `postmortem_summary` — One paragraph suitable for a postmortem document.

Respond ONLY with a JSON object. Do not add any prose or text outside the JSON.
"""

# Friendly JSON template for remediation plan output — avoids triggering Qwen
# tool-call wrapping that occurs when formal Pydantic JSON schemas are injected.
_PLAN_JSON_TEMPLATE = """\
{
  "root_cause": "<single paragraph citing exact metric values and log lines from the telemetry above>",
  "steps": [
    {
      "order": 1,
      "action": "<human-readable action>",
      "command": "<exact kubectl or shell command, or null>",
      "risk": "<low | medium | high>"
    }
  ],
  "rollback_command": "<single command to fully revert all changes>",
  "estimated_mttr_minutes": <integer>,
  "postmortem_summary": "<one paragraph for the postmortem document>"
}"""


async def plan_node(state: dict) -> dict:
    """
    Synthesise all gathered telemetry into a typed RemediationPlan.

    Uses native JSON mode (`bind(response_format={"type": "json_object"})`) and
    a friendly template string rather than a formal Pydantic JSON schema. This
    avoids Groq 400 tool_use_failed errors that occur when Qwen-32b interprets
    a formal schema as a function/tool definition and wraps its output in a
    tool-call envelope.

    The zero-trust guardrail (RemediationPlan.is_high_risk) is evaluated
    here. High-risk plans are logged but not blocked at this stage; blocking
    occurs in the orchestrator's conditional edge before the approval node.

    Reads:
        state["triage_result"] — For service / severity.
        state["raw_alert"]     — For alert title.
        state["telemetry"]     — Full investigation telemetry string.

    Writes:
        state["plan"]             — Markdown-serialised plan for CLI / Rich.
        state["remediation_plan"] — Typed RemediationPlan Pydantic instance.
        state["messages"]         — Appends HumanMessage (input) + AIMessage.
    """
    triage: TriageResult | None = state.get("triage_result")
    service: str = triage.service if triage else state.get("raw_alert", {}).get("service", "unknown")
    severity: str = state.get("severity", "P1")
    alert_title: str = state.get("raw_alert", {}).get("title", "Unknown incident")
    extracted_evidence: str = state.get("extracted_evidence", "No evidence extracted.")

    logger.info("[plan_node] Drafting remediation plan for service=%s", service)

    # Bind JSON mode — do NOT inject a formal JSON schema; Qwen treats formal
    # schemas as function definitions and wraps responses in a tool-call
    # envelope, causing Groq to return 400 tool_use_failed.
    llm = _make_llm().bind(response_format={"type": "json_object"})

    prompt = (
        f"{_PLAN_PROMPT_TEMPLATE.format(service=service, severity=severity, alert_title=alert_title, extracted_evidence=extracted_evidence)}\n\n"
        f"Respond ONLY with a JSON object matching this exact template:\n"
        f"```\n{_PLAN_JSON_TEMPLATE}\n```"
    )
    human_msg = HumanMessage(content=prompt)

    response = await llm.ainvoke([human_msg])
    data = parse_json_robust(response.content)
    plan = RemediationPlan.model_validate(data)

    # Zero-trust guardrail evaluation
    if plan.is_high_risk:
        logger.warning(
            "[plan_node] HIGH-RISK plan generated: rollback_command is empty. "
            "Automated execution will be blocked."
        )
    else:
        logger.info(
            "[plan_node] Plan generated with rollback_command=%r",
            plan.rollback_command[:60],
        )

    # Serialise to markdown for the Rich CLI renderer and approval node
    plan_md = _render_plan_markdown(plan, service, severity)

    ai_msg = AIMessage(content=f"Remediation plan drafted.\n\n{plan_md}")

    return {
        "plan": plan_md,
        "remediation_plan": plan,
        "messages": [human_msg, ai_msg],
    }


# ---------------------------------------------------------------------------
# Private rendering helper
# ---------------------------------------------------------------------------


def _render_plan_markdown(
    plan: RemediationPlan, service: str, severity: str
) -> str:
    """
    Convert a RemediationPlan into a Rich-renderable markdown string.
    This is the exact string surfaced to the operator at the HITL pause.
    """
    risk_icon = "🔴" if plan.is_high_risk else "🟢"
    lines = [
        f"# Remediation Plan — {service} ({severity})",
        "",
        f"**Risk status**: {risk_icon} {'HIGH-RISK (no rollback)' if plan.is_high_risk else 'Standard (rollback available)'}",
        "",
        "## Root Cause",
        plan.root_cause,
        "",
        "## Remediation Steps",
    ]

    for step in sorted(plan.steps, key=lambda s: s.order):
        risk_badge = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(step.risk, "⚪")
        lines.append(f"\n### Step {step.order}: {step.action} {risk_badge}")
        if step.command:
            lines.append(f"```bash\n{step.command}\n```")

    lines += [
        "",
        "## Rollback Command",
        f"```bash\n{plan.rollback_command or '# ⚠ No rollback command provided'}\n```",
        "",
    ]

    if plan.estimated_mttr_minutes:
        lines.append(f"**Estimated MTTR**: {plan.estimated_mttr_minutes} minutes")

    if plan.postmortem_summary:
        lines += ["", "## Postmortem Summary (draft)", plan.postmortem_summary]

    return "\n".join(lines)
