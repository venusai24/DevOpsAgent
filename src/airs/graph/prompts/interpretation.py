"""
Interpretation & Decision Prompts — Module 1.9.

Dynamic prompts assembled at runtime for each investigation hop.
These are assembled from:
  - The current investigation state (alert, hypotheses, evidence context)
  - The retrieved playbook content
  - The current risk state (missing mass, trajectory risk, M_t)

Two-stage prompt architecture:
  Stage A: interpret_evidence   — LLM interprets new tool output
  Stage B: decide_next_action   — LLM decides what to do next (EXECUTE_TOOL etc.)
"""
from __future__ import annotations

from typing import Any


def build_interpret_evidence_prompt(
    alert_name: str,
    alert_description: str,
    service: str,
    hop_index: int,
    tool_name: str,
    server_id: str,
    filtered_content: dict[str, Any],
    leading_hypothesis: str,
    active_evidence_summary: str,
) -> str:
    """
    Stage A prompt: interpret the pre-filtered output of a tool call.

    The LLM is asked to produce a structured interpretation of the evidence
    — what it means for the incident, what it confirms or contradicts.

    Returns:
        Formatted prompt string (injected as user turn in LLM call).
    """
    content_str = _format_content(filtered_content)

    return f"""## Incident Context
Alert: {alert_name}
Description: {alert_description}
Affected Service: {service}
Investigation Hop: {hop_index}

## Tool Output to Interpret
Tool: {server_id}/{tool_name}
Pre-filtered output:
```json
{content_str}
```

## Current Evidence Summary
{active_evidence_summary}

## Leading Hypothesis
{leading_hypothesis or "No hypothesis established yet."}

## Your Task
Interpret this tool output in the context of the incident above.
Provide a structured interpretation covering:
1. What anomaly or finding (if any) does this evidence show?
2. Does this support, contradict, or is it neutral to the leading hypothesis?
3. What new questions does this evidence raise?
4. What is the confidence delta? (positive = more confident in hypothesis, negative = contradicts)

Respond in JSON:
{{
  "finding": "<one-sentence summary of what the evidence shows>",
  "hypothesis_support": "SUPPORTS" | "CONTRADICTS" | "NEUTRAL",
  "confidence_delta": <float between -0.5 and 0.5>,
  "new_questions": ["<question 1>", "<question 2>"],
  "anomalies_detected": [
    {{"metric_or_field": "<name>", "value": "<value>", "significance": "<why it matters>"}}
  ]
}}"""


def build_decide_next_action_prompt(
    alert_name: str,
    service: str,
    namespace: str,
    hop_index: int,
    max_hops: int,
    missing_mass: float,
    trajectory_risk: float,
    supermartingale: float,
    escalation_threshold: float,
    lambda_threshold: float,
    hypotheses_summary: str,
    context_window: dict[str, Any],
    playbook_context: str,
    available_tools: list[dict],
    consecutive_low_delta: int,
) -> str:
    """
    Stage B prompt: decide the next action based on the full investigation state.

    This is the core reasoning step. The LLM sees:
    - Current risk state (missing mass, M_t, trajectory risk)
    - Active evidence (causal chain + other evidence)
    - Hypotheses and their confidence scores
    - Available tools and their signal types
    - Retrieved playbook guidance

    Returns:
        Formatted prompt string.
    """
    tools_str = _format_tools(available_tools)
    causal_chain = context_window.get("causal_chain", [])
    active_evidence = context_window.get("active_evidence", [])
    summarized = context_window.get("summarized_evidence", [])

    causal_str = _format_evidence_list(causal_chain, "Causal Chain")
    active_str = _format_evidence_list(active_evidence, "Active Evidence")
    summarized_str = _format_summarized(summarized)

    hops_remaining = max_hops - hop_index
    pressure_warning = (
        "\n⚠️  ESCALATION ALERT: Supermartingale M_t is near/above 1/δ threshold. "
        "Consider ESCALATE action."
        if supermartingale >= escalation_threshold * 0.80
        else ""
    )
    plateau_warning = (
        f"\n⚠️  PLATEAU DETECTED: Missing mass has not decreased for "
        f"{consecutive_low_delta} consecutive hops. Consider QUERY_PLAYBOOK."
        if consecutive_low_delta >= 2
        else ""
    )

    return f"""## Investigation Status
Alert: {alert_name} | Service: {service} | Namespace: {namespace}
Hop: {hop_index}/{max_hops} ({hops_remaining} hops remaining)

## Risk State
- Missing Mass M_t(λ): {missing_mass:.3f} (target < 0.05 to DIAGNOSE)
- Trajectory Risk R_traj: {trajectory_risk:.3f} (ABSTAIN if next step_risk > {lambda_threshold})
- Supermartingale M_t: {supermartingale:.3f} (ESCALATE if M_t >= {escalation_threshold:.1f})
{pressure_warning}{plateau_warning}

## Hypotheses
{hypotheses_summary}

## Investigation Graph
{causal_str}

{active_str}

{summarized_str}

## Retrieved Playbook Guidance
{playbook_context or "No playbook context available."}

## Available Tools
{tools_str}

## Decision Rules
1. If missing_mass < 0.05 AND leading hypothesis confidence > 0.85 → DIAGNOSE
2. If supermartingale M_t >= {escalation_threshold:.1f} → ESCALATE
3. If consecutive_low_delta >= 3 → QUERY_PLAYBOOK first, then resume tool execution
4. If you have enough evidence to confirm root cause → DIAGNOSE
5. If you need more evidence → EXECUTE_TOOL (you may select up to 3 independent tools to execute in parallel)

## Your Task
Decide the single best next action. Respond ONLY with valid JSON matching this schema:
{{
  "action": "EXECUTE_TOOL" | "QUERY_PLAYBOOK" | "DIAGNOSE" | "ESCALATE",
  "reasoning": "<2-3 sentence justification referencing specific evidence node IDs>",
  "missing_mass_at_decision": {missing_mass:.3f},
  "tool_specs": [
    {{
      "tool_name": "<tool name>",
      "mcp_server_id": "<server id>",
      "arguments": {{}},
      "tier": 1,
      "signal_type": "METRICS" | "LOGS" | "TRACES" | "K8S_STATE",
      "idempotent": true,
      "expected_latency_seconds": 10
    }}
  ],
  "playbook_query": {{
    "collection": "<collection name>",
    "query_text": "<natural language query>",
    "filters": {{}},
    "k": 3
  }}
}}
Note: Include tool_specs only if action=EXECUTE_TOOL; include playbook_query only if action=QUERY_PLAYBOOK."""


def build_diagnosis_prompt(
    alert_name: str,
    service: str,
    namespace: str,
    hop_index: int,
    leading_hypothesis: str,
    causal_chain_summary: str,
    hypotheses_summary: str,
    trajectory_risk: float,
    missing_mass: float,
) -> str:
    """
    Final diagnosis prompt — produce the structured root cause report.
    Called only when transitioning to DIAGNOSE state.
    """
    return f"""## Final Diagnosis Request
Alert: {alert_name} | Service: {service} | Namespace: {namespace}
Investigation completed at hop {hop_index}.

## Confirmed Hypothesis
{leading_hypothesis}

## Causal Chain Evidence
{causal_chain_summary}

## All Hypotheses Considered
{hypotheses_summary}

## Risk Summary
- Final trajectory risk: {trajectory_risk:.3f}
- Final missing mass: {missing_mass:.3f}

## Your Task
Produce the final root cause analysis report as structured JSON:
{{
  "root_cause": "<precise, actionable description of the root cause>",
  "causal_chain": [
    {{"hop": <int>, "entity": "<entity>", "finding": "<what was found>", "evidence_node_id": "<id>"}}
  ],
  "confidence": <float 0-1>,
  "blast_radius": {{
    "affected_services": ["<service1>"],
    "estimated_user_impact": "<description>"
  }},
  "recommended_actions": [
    {{"priority": 1, "action": "<action>", "tool": "<kubectl/runbook ref>", "risk_level": "low|medium|high"}}
  ],
  "escalation_needed": false,
  "investigation_metadata": {{
    "total_hops": {hop_index},
    "trajectory_risk": {trajectory_risk:.3f},
    "final_missing_mass": {missing_mass:.3f}
  }}
}}"""


def build_escalation_prompt(
    alert_name: str,
    service: str,
    hop_index: int,
    escalation_reason: str,
    supermartingale: float,
    trajectory_risk: float,
    leading_hypothesis: str,
    evidence_summary: str,
) -> str:
    """Escalation handoff prompt — produce structured escalation report for human ops."""
    return f"""## Escalation Report
Alert: {alert_name} | Service: {service}
Escalation triggered at hop {hop_index}.

## Reason for Escalation
{escalation_reason}

## Risk Metrics
- Supermartingale M_t: {supermartingale:.3f}
- Trajectory Risk: {trajectory_risk:.3f}

## Best Available Hypothesis
{leading_hypothesis or "No hypothesis established."}

## Evidence Gathered So Far
{evidence_summary}

## Task
Produce a structured escalation handoff for the on-call engineer:
{{
  "escalation_summary": "<2-3 sentence summary of what was investigated and why escalation is needed>",
  "current_best_hypothesis": "<leading hypothesis with confidence>",
  "evidence_gathered": [
    {{"signal": "<signal type>", "entity": "<entity>", "finding": "<key finding>"}}
  ],
  "suggested_next_steps": ["<step 1>", "<step 2>"],
  "urgency": "critical" | "high" | "medium",
  "requires_runbook": "<runbook name or null>"
}}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_content(content: dict) -> str:
    import json
    try:
        return json.dumps(content, indent=2)[:2000]  # Cap at 2K chars
    except Exception:
        return str(content)[:2000]


def _format_tools(tools: list[dict]) -> str:
    if not tools:
        return "No tools available."
    lines = []
    for t in tools[:10]:
        lines.append(
            f"  - {t.get('mcp_server_id', '?')}/{t.get('tool_name', '?')} "
            f"[Tier {t.get('tier', '?')}] — {t.get('signal_type', '?')}"
        )
    return "\n".join(lines)


def _format_evidence_list(nodes: list[dict], label: str) -> str:
    if not nodes:
        return f"### {label}\n(empty)"
    lines = [f"### {label}"]
    for n in nodes[:8]:
        score = n.get("context_score", 0.0)
        entity = n.get("entity", "unknown")
        signal = n.get("signal", "?")
        hop = n.get("hop", "?")
        is_causal = "⭐" if n.get("is_causal") else ""
        node_id = n.get("node_id", "?")[:12]
        lines.append(
            f"  [{node_id}] Hop {hop} | {entity} | {signal} | score={score:.2f} {is_causal}"
        )
    return "\n".join(lines)


def _format_summarized(summaries: list[dict]) -> str:
    if not summaries:
        return "### Summarized Evidence\n(none)"
    lines = ["### Summarized Evidence (compressed)"]
    for s in summaries[:5]:
        lines.append(f"  Hops {s.get('hop_range', '?')}: {s.get('summary', '')}")
    return "\n".join(lines)
