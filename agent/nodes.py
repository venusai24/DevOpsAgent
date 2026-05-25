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
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.state import GraphState, RemediationPlan, RemediationStep, TriageResult, KBRetrievalResult
from agent.tools import ALL_TOOLS, get_logs, get_metrics
from agent.json_parser import parse_json_robust
from agent.playbook_cache import (
    lookup_playbook,
    record_successful_playbook,
    substitute_service,
)
from agent.kb.store import kb_lookup, kb_update_confidence
from agent.kb.token_dedup import deduplicate_telemetry

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

_FAITHFULNESS_WARN_THRESHOLD: float = 0.50
"""
Faithfulness scores below this value indicate the LLM deviated from the
retrieved KB entry. The plan's rollback_command is cleared to force the
zero-trust guardrail to route through reject_node → HITL review.
"""

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
# Node: kb_lookup_node
# ---------------------------------------------------------------------------


async def kb_lookup_node(state: dict) -> dict:
    """
    Knowledge Base lookup gate — runs after triage_node, before any LLM call.

    Performs a hybrid exact/regex/semantic lookup against the KB store and
    writes the result to state["kb_result"]. The orchestrator router then
    reads kb_result to decide whether to:

      bypass_llm=True  (score >= KB_EXACT_BYPASS_THRESHOLD):
        Build RemediationPlan directly from the KB entry, set is_approved=True,
        and route straight to execute_node — skipping investigate, extract,
        plan, and approval entirely.

      score in [KB_RAG_THRESHOLD, KB_EXACT_BYPASS_THRESHOLD):
        Write kb_result to state so plan_node can inject it as RAG context.
        Continue normal investigation pipeline.

      score < KB_RAG_THRESHOLD:
        Write kb_result with match_type='none'. plan_node will operate in
        read-only diagnostic mode (no kubectl commands generated).

    Reads:
        state["raw_alert"]     — For alert title and description.
        state["triage_result"] — For runtime service name substitution.
        state["severity"]      — Preserved verbatim for downstream routing.

    Writes:
        state["kb_result"]       — Always written.
        state["remediation_plan"] — Only on full bypass.
        state["plan"]             — Only on full bypass.
        state["is_approved"]      — Set True only on full bypass.
        state["retry_count"]      — Set to MAX_RETRIES only on full bypass
                                    to signal the router to skip investigation.
        state["messages"]         — One informational AIMessage appended.
    """
    raw_alert: dict = state.get("raw_alert", {})
    alert_title: str = raw_alert.get("title", "")
    alert_description: str = raw_alert.get("description", "")

    triage: TriageResult | None = state.get("triage_result")
    service: str = (
        triage.service if triage else raw_alert.get("service", "unknown")
    )

    logger.info(
        "[kb_lookup_node] Looking up KB for alert=%r service=%s",
        alert_title[:80],
        service,
    )

    kb_result: KBRetrievalResult = await kb_lookup(
        alert_title, alert_description
    )

    updates: dict = {"kb_result": kb_result}

    # ------------------------------------------------------------------
    # Full bypass: exact KB hit — skip the entire LLM pipeline
    # ------------------------------------------------------------------
    if kb_result.bypass_llm and kb_result.entry:
        entry = kb_result.entry

        logger.info(
            "[kb_lookup_node] EXACT BYPASS score=%.3f entry=%s taxonomy=%s "
            "— building plan from KB, skipping LLM pipeline",
            kb_result.retrieval_score,
            entry.entry_id,
            entry.incident_taxonomy,
        )

        # Convert KBRemediationStep → RemediationStep, substituting {service}
        steps = [
            RemediationStep(
                order=s.order,
                action=s.action,
                command=(
                    s.command.replace("{service}", service)
                    if s.command
                    else None
                ),
                risk=s.risk,
            )
            for s in entry.remediation_steps
        ]

        plan = RemediationPlan(
            root_cause=entry.root_cause_narrative,
            steps=steps,
            rollback_command=entry.rollback_command.replace("{service}", service),
            estimated_mttr_minutes=None,
            postmortem_summary=(
                f"[KB-GROUNDED ✔] Entry {entry.entry_id} "
                f"({entry.incident_taxonomy}) matched with score "
                f"{kb_result.retrieval_score:.2f}. "
                f"{entry.root_cause_narrative[:250]}"
            ),
        )

        plan_md = _render_plan_markdown(plan, service, state.get("severity", ""))

        bypass_msg = AIMessage(
            content=(
                f"✅ **KB Exact Bypass** — Entry `{entry.entry_id}` "
                f"(`{entry.incident_taxonomy}`) matched with score "
                f"`{kb_result.retrieval_score:.2f}` "
                f"(threshold `{_settings().KB_EXACT_BYPASS_THRESHOLD}`). "
                f"Executing verified runbook. LLM investigation pipeline skipped."
            )
        )

        updates.update(
            {
                "remediation_plan": plan,
                "plan": plan_md,
                "is_approved": True,   # Full bypass: no HITL for exact KB hits
                "retry_count": MAX_RETRIES,  # Signal router to skip investigation
                "messages": [bypass_msg],
                "faithfulness_score": 1.0,   # KB is the source of truth
            }
        )
    else:
        # RAG match or no match — log and let the pipeline continue
        score_str = f"score={kb_result.retrieval_score:.3f} type={kb_result.match_type}"
        if kb_result.match_type != "none":
            logger.info(
                "[kb_lookup_node] RAG MATCH %s — KB context will be injected into plan_node",
                score_str,
            )
        else:
            logger.info(
                "[kb_lookup_node] NO KB MATCH %s — plan_node will use read-only diagnostic mode",
                score_str,
            )

        info_msg = AIMessage(
            content=(
                f"KB lookup result: {score_str}. "
                + (
                    "Injecting KB context into plan."
                    if kb_result.match_type != "none"
                    else "No KB match — proceeding in read-only diagnostic mode."
                )
            )
        )
        updates["messages"] = [info_msg]

    return updates


def _settings():
    """Lazy import of settings to avoid circular import at module load time."""
    from config import settings
    return settings


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

        # Deduplicate repeated stack traces to prevent context overflow
        new_telemetry = deduplicate_telemetry(
            new_telemetry,
            max_chars=_settings().KB_MAX_TELEMETRY_CHARS,
        )

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

    A token-deduplication pass is applied before the LLM call to guard against
    context-window overflow during cascading failures where the telemetry string
    may contain thousands of repeated stack-trace lines.
    """
    raw_telemetry: str = state.get("telemetry", "No telemetry gathered.")
    cfg = _settings()

    # Guard: compress if over the hard char limit before injecting into LLM
    if len(raw_telemetry) > cfg.KB_MAX_TELEMETRY_CHARS:
        logger.warning(
            "[extract_node] Telemetry %d chars exceeds limit %d — deduplicating",
            len(raw_telemetry),
            cfg.KB_MAX_TELEMETRY_CHARS,
        )
        telemetry = deduplicate_telemetry(raw_telemetry, max_chars=cfg.KB_MAX_TELEMETRY_CHARS)
    else:
        telemetry = raw_telemetry

    logger.info("[extract_node] Extracting key evidence from telemetry (%d chars)", len(telemetry))

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

    RAG Grounding (added for KB robustness):
        If state["kb_result"] contains a match above KB_RAG_THRESHOLD, the
        KB entry's root_cause_narrative and remediation_steps are injected
        into the prompt as "Verified Known Pattern" ground truth. The LLM's
        role shifts from free generator to template instantiator: it fills in
        runtime-specific values (pod names, timestamps) but must NOT invent
        new steps or alter commands.

    Read-only diagnostic mode:
        If no KB match is found (score < KB_RAG_THRESHOLD), the prompt
        explicitly prohibits the LLM from generating kubectl commands.
        All step.command fields will be null, forcing HITL review.

    Faithfulness cross-validation:
        After the LLM returns a plan, _cross_validate_plan() scores how
        closely the generated commands and root cause adhere to the KB entry.
        If faithfulness < _FAITHFULNESS_WARN_THRESHOLD (0.50), the plan's
        rollback_command is cleared, triggering is_high_risk=True and routing
        to reject_node before the HITL approval gate.

    Reads:
        state["triage_result"]    — For service / severity.
        state["raw_alert"]        — For alert title.
        state["extracted_evidence"] — Verbatim evidence from extract_node.
        state["kb_result"]        — KB retrieval result from kb_lookup_node.

    Writes:
        state["plan"]             — Markdown-serialised plan for CLI / Rich.
        state["remediation_plan"] — Typed RemediationPlan Pydantic instance.
        state["faithfulness_score"] — Cross-validation score (None if no KB match).
        state["messages"]         — Appends HumanMessage (input) + AIMessage.
    """
    triage: TriageResult | None = state.get("triage_result")
    service: str = triage.service if triage else state.get("raw_alert", {}).get("service", "unknown")
    severity: str = state.get("severity", "P1")
    alert_title: str = state.get("raw_alert", {}).get("title", "Unknown incident")
    extracted_evidence: str = state.get("extracted_evidence", "No evidence extracted.")
    kb_result: KBRetrievalResult | None = state.get("kb_result")

    cfg = _settings()
    logger.info("[plan_node] Drafting remediation plan for service=%s", service)

    # ------------------------------------------------------------------
    # Build the KB context section for the prompt
    # ------------------------------------------------------------------
    kb_context_section = ""
    read_only_mode = True  # Default to safe read-only until KB says otherwise

    if kb_result and kb_result.entry and kb_result.retrieval_score >= cfg.KB_RAG_THRESHOLD:
        read_only_mode = False
        entry = kb_result.entry
        kb_steps_text = "\n".join(
            [
                f"  {s.order}. [{s.risk.upper()} RISK] {s.action}"
                + (
                    f"\n     Command (`{s.environment}`): `{s.command}`"
                    if s.command
                    else ""
                )
                for s in entry.remediation_steps
            ]
        )
        kb_context_section = (
            f"## ⚠️ Verified Known Pattern (KB Entry `{entry.entry_id}`)"
            f" — GROUND TRUTH\n"
            f"**Taxonomy**: `{entry.incident_taxonomy}`  "
            f"**Confidence**: {entry.confidence_score:.0%}  "
            f"**Match score**: {kb_result.retrieval_score:.2f}\n\n"
            f"**Root Cause (verified)**:\n{entry.root_cause_narrative}\n\n"
            f"**Verified Remediation Steps**:\n{kb_steps_text}\n\n"
            f"**Rollback Command**: `{entry.rollback_command}`\n\n"
            f"⚠️ CRITICAL INSTRUCTION: You MUST derive your `root_cause` and "
            f"`steps` EXCLUSIVELY from the Verified Known Pattern above.\n"
            f"Only substitute runtime-specific values (actual pod names, IP addresses, "
            f"timestamps, transaction IDs) from the Extracted Verbatim Evidence below.\n"
            f"Do NOT invent new steps. Do NOT change command verbs. "
            f"Do NOT add steps not present in the Known Pattern.\n"
        )
        logger.info(
            "[plan_node] KB context injected: entry=%s score=%.3f",
            entry.entry_id, kb_result.retrieval_score,
        )
    else:
        # No KB match — read-only diagnostic mode
        kb_context_section = (
            "## ⚠️ No Verified Pattern Found — READ-ONLY DIAGNOSTIC MODE\n"
            "No matching KB entry was found for this incident pattern.\n\n"
            "❌ CRITICAL INSTRUCTION: You MUST operate in DIAGNOSTIC-ONLY mode:\n"
            "- Set `root_cause` to a diagnostic summary of the available evidence ONLY.\n"
            "- Set ALL `command` fields to `null`. Do NOT suggest any kubectl, shell, or database commands.\n"
            "- Set `rollback_command` to an empty string.\n"
            "A human SRE must review and determine the remediation manually.\n"
        )
        logger.info("[plan_node] No KB match — operating in read-only diagnostic mode")

    # ------------------------------------------------------------------
    # Build the full prompt
    # ------------------------------------------------------------------
    prompt = (
        f"{_PLAN_PROMPT_TEMPLATE.format(service=service, severity=severity, alert_title=alert_title, extracted_evidence=extracted_evidence)}\n\n"
        f"{kb_context_section}\n"
        f"Respond ONLY with a JSON object matching this exact template:\n"
        f"```\n{_PLAN_JSON_TEMPLATE}\n```"
    )
    human_msg = HumanMessage(content=prompt)

    # Bind JSON mode
    llm = _make_llm().bind(response_format={"type": "json_object"})
    response = await llm.ainvoke([human_msg])
    data = parse_json_robust(response.content)
    plan = RemediationPlan.model_validate(data)

    # ------------------------------------------------------------------
    # Faithfulness cross-validation
    # ------------------------------------------------------------------
    faithfulness_score: float | None = None

    if kb_result and kb_result.entry and not read_only_mode:
        faithfulness_score = _cross_validate_plan(plan, kb_result.entry)
        logger.info("[plan_node] Faithfulness score: %.3f", faithfulness_score)

        if faithfulness_score < _FAITHFULNESS_WARN_THRESHOLD:
            logger.warning(
                "[plan_node] LOW-FAITHFULNESS hallucination flag "
                "(score=%.3f < threshold=%.2f) — "
                "clearing rollback_command to force HITL review via reject_node",
                faithfulness_score,
                _FAITHFULNESS_WARN_THRESHOLD,
            )
            plan = RemediationPlan(
                root_cause=plan.root_cause,
                steps=plan.steps,
                rollback_command="",  # Clears is_high_risk → True → reject_node
                estimated_mttr_minutes=plan.estimated_mttr_minutes,
                postmortem_summary=(
                    f"[FAITHFULNESS WARNING: score={faithfulness_score:.2f}] "
                    + plan.postmortem_summary
                ),
            )

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

    plan_md = _render_plan_markdown(plan, service, severity)
    ai_msg = AIMessage(content=f"Remediation plan drafted.\n\n{plan_md}")

    return {
        "plan": plan_md,
        "remediation_plan": plan,
        "faithfulness_score": faithfulness_score,
        "messages": [human_msg, ai_msg],
    }


# ---------------------------------------------------------------------------
# Private: KB faithfulness cross-validation
# ---------------------------------------------------------------------------


def _cross_validate_plan(plan: RemediationPlan, kb_entry) -> float:
    """
    Compute a faithfulness score (0.0–1.0) measuring how closely the LLM-
    generated plan adheres to the retrieved KB entry.

    Two equal-weight components:

    Command fidelity (50%):
        What fraction of the plan's generated commands appear in the KB's
        remediation steps (substring or reverse-substring match after
        normalisation to remove volatile tokens like pod names / IPs)?
        A plan that invents new kubectl verbs not in the KB scores 0.0 here.

    Root cause alignment (50%):
        Jaccard overlap of key terms (length >= 5) between the KB root
        cause narrative and the LLM's root_cause string.  A plan whose
        root_cause explanation shares less than 20% vocabulary with the
        KB entry's narrative is likely hallucinated.

    Args:
        plan:     The RemediationPlan generated by the LLM.
        kb_entry: The KBEntry retrieved from the KB store (type: KBEntry).

    Returns:
        Float in [0.0, 1.0].  Values < 0.50 trigger the hallucination flag.
    """
    # --- Command fidelity ---
    kb_commands = [s.command for s in kb_entry.remediation_steps if s.command]
    gen_commands = [s.command for s in plan.steps if s.command]

    if not kb_commands:
        command_fidelity = 1.0  # No KB commands to compare against; not a signal of drift
    elif not gen_commands:
        command_fidelity = 0.0  # LLM generated no commands despite KB providing them
    else:
        grounded_count = sum(
            1
            for gc in gen_commands
            if any(
                _normalise_cmd(kc) in _normalise_cmd(gc)
                or _normalise_cmd(gc) in _normalise_cmd(kc)
                for kc in kb_commands
            )
        )
        command_fidelity = grounded_count / len(gen_commands)

    # --- Root cause alignment ---
    kb_terms = set(re.findall(r"\b[a-z]{5,}\b", kb_entry.root_cause_narrative.lower()))
    gen_terms = set(re.findall(r"\b[a-z]{5,}\b", plan.root_cause.lower()))

    if not kb_terms:
        root_alignment = 1.0
    else:
        overlap = len(kb_terms & gen_terms) / len(kb_terms | gen_terms)
        # Scale: Jaccard of 0.15+ on domain-specific SRE vocabulary is strong
        root_alignment = min(1.0, overlap * 2.0)

    score = 0.5 * command_fidelity + 0.5 * root_alignment
    logger.debug(
        "[_cross_validate_plan] cmd_fidelity=%.3f root_alignment=%.3f final=%.3f",
        command_fidelity, root_alignment, score,
    )
    return round(score, 4)


def _normalise_cmd(cmd: str) -> str:
    """
    Normalise a shell / kubectl command for similarity comparison by stripping
    volatile runtime tokens: pod name suffixes, IP addresses, hex values,
    and Kubernetes-style random suffixes.
    """
    # Remove Kubernetes pod name suffixes: -7d9f8b-xkzp2
    normalised = re.sub(r"-[a-z0-9]{5,10}-[a-z0-9]{5}\b", "", cmd)
    # Remove IP addresses
    normalised = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<ip>", normalised)
    # Remove hex values
    normalised = re.sub(r"0x[0-9a-fA-F]+", "<hex>", normalised)
    # Remove standalone integers (ports, counts, memory sizes)
    normalised = re.sub(r"\b\d{3,}\b", "<n>", normalised)
    # Collapse whitespace and lowercase
    return re.sub(r"\s+", " ", normalised).strip().lower()


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
