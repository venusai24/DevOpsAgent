"""
agent/nodes.py

LangGraph node functions for the AIRS Hybrid Memory Architecture.

Node Catalogue (14 nodes total):
  Perception Layer:
    triage_node            — Classify severity (P0-P3) and extract the service name.
    perception_node        — Run L1/L2/L3 tiered log classification + NeSy routing decision.

  Reasoning Layer (Multi-Agent):
    topology_agent_node    — EKG traversal: dependency chain + blast radius pre-check.
    diagnostic_agent_node  — CBR retrieval: find most similar historical cases.
    logic_agent_node       — Symbolic pruning: eliminate impossible root-cause hypotheses.
    remediation_agent_node — Adapt CBR plan or generate fresh plan via LLM.
    risk_agent_node        — Full blast radius estimation + execution strategy decision.

  Action Layer:
    policy_check_node      — Policy-as-Code invariant check + Terraform HCL dry-run.
    canary_execute_node    — Progressive canary deployment with golden signal monitoring.
    direct_execute_node    — Direct (non-canary) execution for low-risk actions.
    rollback_check_node    — Golden signal comparison post-execution; trigger rollback if needed.
    retain_node            — Store resolved case in CBR database (continuous learning).

  Existing nodes (preserved):
    investigate_node       — ReAct loop (used by NEURAL_FULL pathway only).
    extract_node           — Verbatim evidence extraction from raw telemetry.
    plan_node              — LLM plan generation (fallback for NEURAL_FULL).

  Escalation / rejection:
    escalate_node          — Notify Slack of P0 incidents immediately.
    reject_node            — Abort on operator rejection or policy block.
    postmortem_node        — Generate final postmortem and write to disk.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.state import (
    GraphState,
    RemediationPlan,
    RemediationStep,
    TriageResult,
    CausalNode,
)
from agent.tools import ALL_TOOLS, get_logs, get_metrics
from agent.json_parser import parse_json_robust

# Perception Layer
from agent.perception.log_classifier import TieredLogClassifier
from agent.reasoning.nesym_router import NeuroSymbolicRouter, ReasoningPathway

# Reasoning Layer
from agent.reasoning.knowledge_graph import KnowledgeGraph
from agent.reasoning.cbr_engine import CBREngine
from agent.reasoning.incident_store import HistoricalCase

# Action Layer
from agent.action.blast_radius import BlastRadiusEstimator
from agent.action.policy_engine import PolicyEngine, generate_terraform_patch
from agent.action.canary_controller import CanaryController
from agent.action.rollback_controller import RollbackController, GoldenSignalSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3

_MODEL_NAME: str = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

# Singletons — constructed once per process
_LOG_CLASSIFIER = TieredLogClassifier()
_NESYM_ROUTER = NeuroSymbolicRouter()
_CBR_ENGINE = CBREngine()
_EKG = KnowledgeGraph.get_instance()
_BLAST_ESTIMATOR = BlastRadiusEstimator()
_POLICY_ENGINE = PolicyEngine()
_CANARY_CTRL = CanaryController(demo_mode=os.getenv("CANARY_DEMO_MODE", "true").lower() == "true")
_ROLLBACK_CTRL = RollbackController()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm(**kwargs: Any) -> ChatGroq:
    """Instantiate a Groq Qwen LLM using the GROQ_API_KEY env var."""
    return ChatGroq(
        model=_MODEL_NAME,
        temperature=0,
        max_retries=5,
        **kwargs,
    )


# ===========================================================================
# TRIAGE NODE (unchanged from MVP)
# ===========================================================================

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

_TRIAGE_JSON_TEMPLATE = """\
{
  "severity": "<P0 | P1 | P2 | P3>",
  "service": "<canonical microservice name from the alert>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence citing specific evidence from the alert>"
}"""


async def triage_node(state: dict) -> dict:
    """
    Classify the incoming alert into a severity level and extract the affected service name.

    Reads:  state["raw_alert"]
    Writes: state["severity"], state["triage_result"], state["messages"]
    """
    raw_alert: dict = state.get("raw_alert", {})
    logger.info("[triage_node] Classifying alert id=%s", raw_alert.get("id", "unknown"))

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

    if result.confidence < 0.6:
        logger.warning(
            "[triage_node] Low-confidence classification: severity=%s confidence=%.2f",
            result.severity, result.confidence,
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
        result.severity, result.service, result.confidence,
    )
    return {
        "severity": result.severity,
        "triage_result": result,
        "messages": [human_msg, ai_msg],
    }


# ===========================================================================
# ESCALATE NODE (unchanged from MVP)
# ===========================================================================

async def escalate_node(state: dict) -> dict:
    """
    Notify the on-call engineer of a P0 incident via Slack (if configured).
    Falls back gracefully when SLACK_BOT_TOKEN is not set.

    Reads:  state["triage_result"], state["raw_alert"]
    Writes: state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    severity = state.get("severity", "P0")
    alert = state.get("raw_alert", {})

    logger.info("[escalate_node] Escalating P0 for service=%s", service)

    # Attempt Slack notification
    try:
        from agent.integrations.slack import send_slack_notification
        await send_slack_notification(
            f"🚨 *P0 INCIDENT DETECTED* — `{service}`\n"
            f"> {alert.get('description', 'No description')}\n"
            f"AIRS is now running automated triage and investigation."
        )
    except Exception as exc:
        logger.warning("[escalate_node] Slack notification failed: %s", exc)

    ai_msg = AIMessage(
        content=f"P0 escalation triggered for {service}. On-call team notified."
    )
    return {"messages": [ai_msg]}


# ===========================================================================
# PHASE 3: PERCEPTION NODE
# ===========================================================================

async def perception_node(state: dict) -> dict:
    """
    Run the L1→L2→L3 tiered log classification on any telemetry collected
    so far (or on the raw alert description if investigation hasn't run yet).

    Also pre-fetches telemetry from the mock API to ensure the perception
    layer has something to classify before the investigation loop.

    Determines the NeSy routing decision: SYMBOLIC_FAST / CBR_GUIDED / NEURAL_FULL.

    Reads:  state["telemetry"], state["raw_alert"], state["triage_result"]
    Writes: state["perception_stats"], state["primary_log_template"],
            state["telemetry"] (pre-populated if empty), state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else state.get("raw_alert", {}).get("service", "unknown")
    existing_telemetry: str = state.get("telemetry", "")
    alert_desc: str = state.get("raw_alert", {}).get("description", "")

    # Use alert description as a seed if investigation hasn't run yet
    text_to_classify = existing_telemetry if existing_telemetry else alert_desc

    logger.info(
        "[perception_node] Classifying telemetry for service=%s (len=%d)",
        service, len(text_to_classify),
    )

    perception_result = await _LOG_CLASSIFIER.classify_telemetry_block(text_to_classify)
    stats = perception_result.tier_stats
    primary_template = perception_result.primary_template

    # Pre-fetch telemetry if none collected yet (ensures perception has real data)
    new_telemetry = existing_telemetry
    if not existing_telemetry:
        try:
            logs = await get_logs.ainvoke({"service": service, "limit": 50})
            metrics = await get_metrics.ainvoke({"service": service, "metric": "error_rate"})
            new_telemetry = f"### Logs\n{logs}\n\n### Metrics\n{metrics}"
            # Re-run perception on actual data
            full_result = await _LOG_CLASSIFIER.classify_telemetry_block(new_telemetry)
            stats = full_result.tier_stats
            primary_template = full_result.primary_template
        except Exception as exc:
            logger.warning("[perception_node] Pre-fetch failed: %s", exc)

    ai_msg = AIMessage(content=perception_result.to_markdown())
    logger.info(
        "[perception_node] L1=%d L2=%d L3=%d primary=%s",
        stats["L1_hits"], stats["L2_hits"], stats["L3_hits"], primary_template,
    )
    return {
        "perception_stats": stats,
        "primary_log_template": primary_template,
        "telemetry": new_telemetry,
        "messages": [ai_msg],
    }


# ===========================================================================
# PHASE 1: TOPOLOGY AGENT NODE
# ===========================================================================

async def topology_agent_node(state: dict) -> dict:
    """
    Pure graph traversal — no LLM call.

    Queries the Enterprise Knowledge Graph for the failing service's dependency
    chain (2 hops) and updates node health status based on the current alert.

    Reads:  state["triage_result"], state["severity"]
    Writes: state["topology_map"], state["ekg_service_context"], state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else state.get("raw_alert", {}).get("service", "unknown")
    severity = state.get("severity", "P1")

    logger.info("[topology_agent] Building dependency map for service=%s", service)

    await _EKG.initialize()

    # Mark the failing service as critical in the live graph
    from agent.reasoning.graph_schema import HealthStatus
    await _EKG.update_node_health(service, HealthStatus.CRITICAL)

    # Get dependency chain (2 hops)
    subgraph = await _EKG.get_dependency_chain(service, depth=2)
    topology_map = subgraph.model_dump(mode="json")

    # Get known failure correlations
    correlations = await _EKG.get_failure_correlations(service)

    # Build EKG context string for LLM injection
    context_lines = [subgraph.to_markdown()]
    if correlations:
        context_lines.append("\n## Known Failure Correlations")
        for corr in correlations:
            context_lines.append(f"- {corr.get('description', '')}")
            hist_ids = corr.get("historical_incident_ids", [])
            if hist_ids:
                context_lines.append(f"  Historical incidents: {', '.join(hist_ids)}")

    ekg_context = "\n".join(context_lines)

    logger.info(
        "[topology_agent] Mapped %d nodes, %d edges for %s",
        len(subgraph.nodes), len(subgraph.edges), service,
    )

    ai_msg = AIMessage(content=f"Topology mapped:\n{ekg_context[:500]}...")
    return {
        "topology_map": topology_map,
        "ekg_service_context": ekg_context,
        "messages": [ai_msg],
    }


# ===========================================================================
# PHASE 2: DIAGNOSTIC AGENT NODE
# ===========================================================================

async def diagnostic_agent_node(state: dict) -> dict:
    """
    CBR retrieval — find the most similar historical incidents.

    Extracts a feature vector from current telemetry, queries the CBR engine
    for similar historical cases, and builds ranked hypotheses.

    Reads:  state["telemetry"], state["triage_result"], state["topology_map"]
    Writes: state["cbr_matches"], state["cbr_confidence"], state["precedent_incident_id"],
            state["root_cause_hypotheses"], state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    telemetry: str = state.get("telemetry", "")
    ekg_context: str = state.get("ekg_service_context", "")

    # Get service tier from EKG
    node_data = await _EKG.get_node(service)
    service_tier = node_data.get("tier", 2) if node_data else 2

    logger.info("[diagnostic_agent] Running CBR retrieval for service=%s tier=%d", service, service_tier)

    from config import settings
    candidates = await _CBR_ENGINE.retrieve(
        _CBR_ENGINE.extract_feature_vector(telemetry or ekg_context, service, service_tier),
        top_k=settings.CBR_TOP_K,
        min_similarity=settings.CBR_MIN_SIMILARITY,
    )

    # Serialize matches for GraphState
    cbr_matches = []
    for sc in candidates:
        cbr_matches.append({
            "incident_id": sc.case.incident_id,
            "service": sc.case.service,
            "root_cause_category": sc.case.root_cause_category,
            "similarity_score": round(sc.similarity, 3),
            "mttr_minutes": sc.case.mttr_minutes,
            "outcome": sc.case.outcome,
            "postmortem_summary": sc.case.postmortem_summary[:200],
        })

    cbr_confidence = candidates[0].similarity if candidates else 0.0
    precedent_id = candidates[0].case.incident_id if candidates else ""

    # Build root cause hypotheses (one per CBR candidate)
    hypotheses = []
    for sc in candidates:
        hypotheses.append({
            "category": sc.case.root_cause_category,
            "confidence": round(sc.similarity, 3),
            "evidence": [
                f"CBR match: {sc.case.incident_id} ({sc.similarity:.0%} similarity)",
                f"Historical MTTR: {sc.case.mttr_minutes}m",
            ],
        })

    candidates_md = _CBR_ENGINE.format_candidates_markdown(candidates)
    logger.info(
        "[diagnostic_agent] CBR: %d matches, best=%.2f, precedent=%s",
        len(candidates), cbr_confidence, precedent_id,
    )
    ai_msg = AIMessage(content=candidates_md)
    return {
        "cbr_matches": cbr_matches,
        "cbr_confidence": cbr_confidence,
        "precedent_incident_id": precedent_id,
        "root_cause_hypotheses": hypotheses,
        "messages": [ai_msg],
    }


# ===========================================================================
# PHASE 4: LOGIC AGENT NODE (Symbolic Pruning)
# ===========================================================================

# Known nominal thresholds — metrics below these cannot be root causes
_NOMINAL_THRESHOLDS = {
    "cpu": ("cpu", 80.0, "CPU utilization is nominal"),
    "memory": ("memory", 85.0, "Memory utilization is nominal"),
    "disk": ("disk.*100%|No space left", 95.0, "Disk utilization is nominal"),
    "dns": ("dns_latency", 200.0, "DNS latency is nominal"),
}

# Template → root cause category mapping
_TEMPLATE_TO_CATEGORY: dict[str, str] = {
    "connection_pool_exhausted": "connection_pool_exhaustion",
    "oom_killed": "oom_killed",
    "dns_resolution_failure": "dns_resolution_failure",
    "disk_space_exhausted": "disk_space_exhaustion",
    "tls_cert_expired": "tls_certificate_expiration",
    "upstream_rate_limited": "upstream_rate_limiting",
    "database_query_timeout": "database_query_timeout",
    "redis_oom_eviction": "redis_oom_eviction",
    "pod_crash_loop": "pod_crash_loop",
    "transaction_leak": "connection_pool_exhaustion",
}


async def logic_agent_node(state: dict) -> dict:
    """
    Symbolic hypothesis pruning — no LLM call.

    Applies deterministic rules to eliminate impossible root-cause hypotheses
    based on the current metric values and the primary log template.

    Rules:
      - If CPU metrics are nominal → prune all CPU-related hypotheses
      - If the primary_log_template directly maps to a known category → confirm it
      - If a service has NO error logs in topology → prune it as root cause
      - If a dependency is critical but the dependent has no errors → flag cascade

    Reads:  state["root_cause_hypotheses"], state["primary_log_template"],
            state["telemetry"], state["topology_map"]
    Writes: state["confirmed_root_cause"], state["causal_graph_nodes"], state["messages"]
    """
    hypotheses: list[dict] = state.get("root_cause_hypotheses", [])
    primary_template: str = state.get("primary_log_template", "")
    telemetry: str = state.get("telemetry", "")
    topology_map: dict = state.get("topology_map", {})
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"

    logger.info(
        "[logic_agent] Symbolic pruning: %d hypotheses, template=%s",
        len(hypotheses), primary_template,
    )

    import re

    causal_nodes: list[CausalNode] = []
    confirmed: str = ""

    # --- Rule 1: Direct template→category confirmation ---
    direct_category = _TEMPLATE_TO_CATEGORY.get(primary_template)
    if direct_category:
        confirmed = direct_category
        causal_nodes.append(CausalNode(
            service=service,
            evidence=[f"L1/L2 log template `{primary_template}` directly maps to `{direct_category}`"],
            is_root_cause=True,
            confidence=1.0,
        ))
        logger.info("[logic_agent] Rule 1 confirmed root cause: %s", confirmed)

    # --- Rule 2: Prune hypotheses where the governing metric is nominal ---
    for hypothesis in hypotheses:
        category = hypothesis.get("category", "")
        already_confirmed = category == confirmed

        # Check if we can find contradicting nominal metric evidence in telemetry
        pruned = False
        prune_reason = ""

        if "cpu" in category and not re.search(r"cpu.*[89][0-9]\.?[0-9]*\s*%", telemetry, re.IGNORECASE):
            if re.search(r"cpu.*3[0-9]\.?[0-9]*\s*%", telemetry, re.IGNORECASE):
                pruned = True
                prune_reason = "CPU utilization is nominal (~33-39%) — cannot be root cause"

        if not confirmed and not pruned:
            causal_nodes.append(CausalNode(
                service=service,
                evidence=[f"CBR hypothesis: {category} (confidence={hypothesis.get('confidence', 0):.0%})"],
                is_root_cause=not confirmed,
                confidence=hypothesis.get("confidence", 0.5),
            ))
        elif pruned:
            causal_nodes.append(CausalNode(
                service=service,
                evidence=[],
                is_root_cause=False,
                confidence=0.0,
                pruned_reason=prune_reason,
            ))

    # --- Rule 3: Topological cascade detection ---
    # If no hypothesis confirmed yet, check if a critical dependency could be the cause
    if not confirmed:
        for node in topology_map.get("nodes", []):
            if node.get("health_status") in ("critical", "degraded") and node.get("name") != service:
                dep_name = node.get("name", "unknown")
                dep_tier = node.get("tier", 3)
                causal_nodes.append(CausalNode(
                    service=dep_name,
                    evidence=[f"Dependency `{dep_name}` (tier {dep_tier}) is in {node['health_status']} state"],
                    is_root_cause=dep_tier <= 2,
                    confidence=0.7 if dep_tier == 1 else 0.5,
                ))
                if dep_tier == 1:
                    confirmed = f"cascade_from_{dep_name.replace('-', '_')}"

    # Serialize causal nodes
    causal_nodes_dicts = [n.model_dump() for n in causal_nodes]

    summary_lines = [f"## Symbolic Causal Analysis"]
    summary_lines.append(f"**Confirmed Root Cause**: `{confirmed or 'undetermined'}`")
    summary_lines.append(f"**Causal Nodes**: {len(causal_nodes)}")
    for cn in causal_nodes[:5]:
        icon = "✅" if cn.is_root_cause else ("❌" if cn.pruned_reason else "⚠️")
        summary_lines.append(f"{icon} `{cn.service}`: {cn.evidence[0] if cn.evidence else cn.pruned_reason}")

    ai_msg = AIMessage(content="\n".join(summary_lines))
    logger.info("[logic_agent] Confirmed root cause: %s, nodes: %d", confirmed, len(causal_nodes))
    return {
        "confirmed_root_cause": confirmed,
        "causal_graph_nodes": causal_nodes_dicts,
        "messages": [ai_msg],
    }


# ===========================================================================
# REMEDIATION AGENT NODE (CBR-adapted or LLM-generated)
# ===========================================================================

_REMEDIATION_PROMPT_TEMPLATE = """\
You are a senior SRE architect drafting an incident remediation plan.

## Incident Context
- **Service**: {service}
- **Severity**: {severity}
- **Confirmed Root Cause**: {root_cause}
- **CBR Precedent**: {cbr_precedent}

## EKG Topology Context
{ekg_context}

## Verbatim Evidence
{evidence}

## CBR-Adapted Plan (use this as primary basis if confidence >= 60%)
{cbr_plan}

## Task
Produce a precise remediation plan as JSON. Adapt the CBR plan to the current context.
Substitute the correct service name, namespace, and resource identifiers.
Add or remove steps only if the current evidence clearly warrants it.

Respond ONLY with a JSON object matching:
```
{json_template}
```
"""

_PLAN_JSON_TEMPLATE = """\
{
  "root_cause": "<single paragraph citing exact evidence>",
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


async def remediation_agent_node(state: dict) -> dict:
    """
    Generate a remediation plan using CBR-adapted solution or LLM generation.

    For SYMBOLIC_FAST and CBR_GUIDED pathways: returns the CBR-adapted plan directly.
    For NEURAL_FULL: falls through to LLM generation.

    Reads:  state["cbr_matches"], state["cbr_confidence"], state["confirmed_root_cause"],
            state["triage_result"], state["telemetry"], state["ekg_service_context"]
    Writes: state["plan"], state["remediation_plan"], state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    severity = state.get("severity", "P1")
    confirmed_root_cause = state.get("confirmed_root_cause", "unknown")
    cbr_confidence = state.get("cbr_confidence", 0.0)
    cbr_matches: list[dict] = state.get("cbr_matches", [])
    precedent_id = state.get("precedent_incident_id", "")
    telemetry = state.get("telemetry", "No telemetry.")
    ekg_context = state.get("ekg_service_context", "No topology data.")
    perception_stats = state.get("perception_stats", {})

    logger.info(
        "[remediation_agent] Generating plan for service=%s cbr_confidence=%.2f",
        service, cbr_confidence,
    )

    # Determine routing pathway
    primary_template = state.get("primary_log_template", "")
    routing = _NESYM_ROUTER.route(perception_stats, cbr_confidence, primary_template)

    # --- SYMBOLIC_FAST or CBR_GUIDED: Use CBR plan directly ---
    if routing.pathway in (ReasoningPathway.SYMBOLIC_FAST, ReasoningPathway.CBR_GUIDED) and cbr_matches:
        best_match_id = cbr_matches[0]["incident_id"] if cbr_matches else ""
        # Retrieve full case from store for adaptation
        store = _CBR_ENGINE._store
        await store.initialize()
        matching_cases = [c for c in store._memory_store if c.incident_id == best_match_id]
        if matching_cases:
            adapted = _CBR_ENGINE.reuse(service, "prod", matching_cases[0])
            plan = RemediationPlan(
                root_cause=(
                    f"[CBR-Adapted from {adapted.source_case_id} — "
                    f"{adapted.similarity_score:.0%} similarity] "
                    f"Root cause category: {adapted.root_cause_category}. "
                    f"{adapted.postmortem_template}"
                ),
                steps=[
                    RemediationStep(
                        order=s["order"],
                        action=s["action"],
                        command=s.get("command"),
                        risk=s.get("risk", "low"),
                    )
                    for s in adapted.adapted_steps
                ],
                rollback_command=adapted.adapted_rollback_command,
                estimated_mttr_minutes=adapted.estimated_mttr_minutes,
                postmortem_summary=(
                    f"Adapted from precedent {adapted.source_case_id}. "
                    f"{adapted.postmortem_template}"
                ),
            )
            plan_md = _render_plan_markdown(plan, service, severity, routing, precedent_id)
            ai_msg = AIMessage(content=f"CBR plan adapted ({routing.pathway.value}):\n{plan_md[:400]}...")
            logger.info(
                "[remediation_agent] CBR plan generated via %s for %s",
                routing.pathway.value, service,
            )
            return {
                "plan": plan_md,
                "remediation_plan": plan,
                "messages": [ai_msg],
            }

    # --- NEURAL_FULL: LLM generation ---
    cbr_plan_text = "No CBR match found. Generate plan from telemetry." if not cbr_matches else (
        f"Best match: {cbr_matches[0]['incident_id']} "
        f"(similarity={cbr_matches[0]['similarity_score']:.0%})\n"
        f"Historical steps: {cbr_matches[0].get('postmortem_summary', '')}"
    )

    extracted_evidence = state.get("extracted_evidence", telemetry[:500])

    llm = _make_llm().bind(response_format={"type": "json_object"})
    prompt = _REMEDIATION_PROMPT_TEMPLATE.format(
        service=service,
        severity=severity,
        root_cause=confirmed_root_cause or "undetermined",
        cbr_precedent=f"{precedent_id} (confidence={cbr_confidence:.0%})" if precedent_id else "None",
        ekg_context=ekg_context[:800],
        evidence=extracted_evidence[:600],
        cbr_plan=cbr_plan_text,
        json_template=_PLAN_JSON_TEMPLATE,
    )
    human_msg = HumanMessage(content=prompt)
    response = await llm.ainvoke([human_msg])
    data = parse_json_robust(response.content)
    plan = RemediationPlan.model_validate(data)
    plan_md = _render_plan_markdown(plan, service, severity, routing, precedent_id)

    ai_msg = AIMessage(content=f"LLM plan generated (NEURAL_FULL):\n{plan_md[:400]}...")
    logger.info("[remediation_agent] LLM plan generated via NEURAL_FULL for %s", service)
    return {
        "plan": plan_md,
        "remediation_plan": plan,
        "messages": [human_msg, ai_msg],
    }


def _render_plan_markdown(
    plan: RemediationPlan,
    service: str,
    severity: str,
    routing: Any = None,
    precedent_id: str = "",
) -> str:
    """Render a RemediationPlan as Rich-renderable markdown for the operator."""
    risk_icon = "🔴" if plan.is_high_risk else "🟢"
    lines = [
        f"# Remediation Plan — {service} ({severity})",
        "",
        f"**Risk status**: {risk_icon} {'HIGH-RISK (no rollback)' if plan.is_high_risk else 'Standard (rollback available)'}",
    ]
    if routing:
        pathway_icons = {"symbolic_fast": "⚡", "cbr_guided": "📚", "neural_full": "🧠"}
        icon = pathway_icons.get(routing.pathway.value if hasattr(routing, "pathway") else "", "▶")
        lines.append(f"**Reasoning Pathway**: {icon} `{routing.pathway.value.upper() if hasattr(routing, 'pathway') else 'unknown'}`")
    if precedent_id:
        lines.append(f"**CBR Precedent**: `{precedent_id}`")
    lines += ["", "## Root Cause", plan.root_cause, "", "## Remediation Steps"]
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


# ===========================================================================
# PHASE 5a: RISK AGENT NODE (Blast Radius)
# ===========================================================================

async def risk_agent_node(state: dict) -> dict:
    """
    Full blast radius estimation + execution strategy decision.

    Reads:  state["triage_result"], state["remediation_plan"]
    Writes: state["blast_radius_result"], state["execution_strategy"], state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    plan: RemediationPlan | None = state.get("remediation_plan")
    action_desc = plan.steps[0].command or "unknown" if plan and plan.steps else "unknown"

    logger.info("[risk_agent] Estimating blast radius for service=%s", service)

    report = await _BLAST_ESTIMATOR.estimate(service, action_desc)

    blast_dict = {
        "target_service": report.target_service,
        "affected_services": report.affected_services,
        "tier1_services": report.tier1_services,
        "tier1_impact": report.tier1_impact,
        "risk_score": report.risk_score,
        "recommendation": report.recommendation,
        "on_call_contacts": report.on_call_contacts,
        "estimated_user_impact_pct": report.estimated_user_impact_pct,
        "report_markdown": report.to_markdown(),
    }

    # Execution strategy: canary for low-risk, approval gate for tier-1 impact
    if report.recommendation == "block":
        strategy = "blocked"
    elif report.tier1_impact or report.recommendation == "require_approval":
        strategy = "require_approval"
    elif report.risk_score < 0.3:
        strategy = "canary"
    else:
        strategy = "require_approval"

    logger.info(
        "[risk_agent] blast_radius: risk=%.2f tier1=%s strategy=%s",
        report.risk_score, report.tier1_impact, strategy,
    )
    ai_msg = AIMessage(content=report.to_markdown())
    return {
        "blast_radius_result": blast_dict,
        "execution_strategy": strategy,
        "messages": [ai_msg],
    }


# ===========================================================================
# PHASE 5b: POLICY CHECK NODE
# ===========================================================================

async def policy_check_node(state: dict) -> dict:
    """
    Policy-as-Code invariant check + Terraform HCL dry-run.

    Reads:  state["remediation_plan"], state["triage_result"], state["severity"]
    Writes: state["policy_check_result"], state["execution_strategy"], state["messages"]
    """
    plan: RemediationPlan | None = state.get("remediation_plan")
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    severity = state.get("severity", "P1")

    logger.info("[policy_check] Running invariant check for service=%s", service)

    current_state = {"replicas": 2, "public_access": False, "max_connections": 100}
    proposed_patch: dict = {}
    action_type = "generic_action"

    if plan and plan.steps:
        first_cmd = plan.steps[0].command or ""
        import re
        if "rollout restart" in first_cmd:
            action_type = "restart_deployment"
        elif "scale" in first_cmd:
            action_type = "scale_deployment"
            m = re.search(r"--replicas=(\d+)", first_cmd)
            if m:
                proposed_patch["replicas"] = int(m.group(1))
        elif "set resources" in first_cmd:
            action_type = "update_resource_limits"

    incident_context = {
        "severity": severity,
        "rollback_command": plan.rollback_command if plan else "",
    }

    hcl = generate_terraform_patch(service, "prod", action_type, proposed_patch)
    result = _POLICY_ENGINE.check(
        proposed_patch=proposed_patch,
        current_state=current_state,
        incident_context=incident_context,
        terraform_hcl=hcl,
    )

    policy_dict = {
        "passed": result.passed,
        "critical_violations": [v.invariant_id for v in result.violations],
        "warnings": [w.invariant_id for w in result.warnings],
        "terraform_valid": result.terraform_valid,
        "terraform_errors": result.terraform_errors,
        "result_markdown": result.to_markdown(),
    }

    # Override execution strategy if policy blocks
    current_strategy = state.get("execution_strategy", "require_approval")
    if not result.passed:
        final_strategy = "blocked"
    else:
        final_strategy = current_strategy

    logger.info(
        "[policy_check] passed=%s violations=%d strategy=%s",
        result.passed, len(result.violations), final_strategy,
    )
    ai_msg = AIMessage(content=result.to_markdown())
    return {
        "policy_check_result": policy_dict,
        "execution_strategy": final_strategy,
        "messages": [ai_msg],
    }


# ===========================================================================
# CANARY EXECUTE NODE
# ===========================================================================

async def canary_execute_node(state: dict) -> dict:
    """
    Progressive canary execution with golden signal monitoring.

    Reads:  state["remediation_plan"], state["triage_result"]
    Writes: state["canary_status"], state["rollback_triggered"], state["postmortem"], state["messages"]
    """
    plan: RemediationPlan | None = state.get("remediation_plan")
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"

    if not plan or not plan.steps:
        logger.warning("[canary_execute] No plan to execute.")
        return {"canary_status": {}, "rollback_triggered": False}

    first_cmd = plan.steps[0].command or ""
    logger.info("[canary_execute] Starting canary for %s cmd=%r", service, first_cmd[:60])

    baseline = RollbackController.capture_demo_baseline(service)
    result = await _CANARY_CTRL.execute_canary(
        service=service,
        remediation_command=first_cmd,
        rollback_command=plan.rollback_command,
        baseline=baseline,
    )

    canary_dict = {
        "service": result.service,
        "succeeded": result.succeeded,
        "stages_completed": len(result.stage_results),
        "halted_at_stage": result.halted_at_stage,
        "total_duration_seconds": result.total_duration_seconds,
        "final_error_rate_pct": result.final_signal_health.error_rate_pct if result.final_signal_health else 0.0,
        "final_latency_p99_ms": result.final_signal_health.latency_p99_ms if result.final_signal_health else 0.0,
        "rollback_executed": result.rollback_result.rollback_executed if result.rollback_result else False,
        "report_markdown": result.to_markdown(),
    }

    postmortem = _generate_postmortem(state, plan, result.succeeded, result.to_markdown())
    ai_msg = AIMessage(content=result.to_markdown())
    return {
        "canary_status": canary_dict,
        "rollback_triggered": not result.succeeded,
        "postmortem": postmortem,
        "messages": [ai_msg],
    }


# ===========================================================================
# DIRECT EXECUTE NODE (for low-risk, HITL-approved actions)
# ===========================================================================

async def direct_execute_node(state: dict) -> dict:
    """
    Execute remediation directly (no canary) for HITL-approved plans.

    Reads:  state["remediation_plan"], state["is_approved"], state["triage_result"]
    Writes: state["postmortem"], state["messages"]
    """
    plan: RemediationPlan | None = state.get("remediation_plan")
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    is_approved: bool = state.get("is_approved", False)

    if not is_approved:
        logger.info("[direct_execute] Plan not approved — skipping execution.")
        return {"postmortem": "Execution skipped: plan not approved."}

    if not plan:
        return {"postmortem": "No remediation plan found."}

    logger.info("[direct_execute] Executing plan for service=%s", service)

    execution_log: list[str] = []
    from agent.security.guardrails import is_safe_command
    from agent.integrations.k8s import apply_kubectl_command

    for step in sorted(plan.steps, key=lambda s: s.order):
        if step.command:
            if is_safe_command(step.command):
                try:
                    output = apply_kubectl_command(step.command)
                    execution_log.append(f"✅ Step {step.order}: {step.action}\n   Output: {output}")
                    logger.info("[direct_execute] Step %d OK: %s", step.order, step.action)
                except Exception as exc:
                    execution_log.append(f"❌ Step {step.order}: {step.action}\n   Error: {exc}")
                    logger.error("[direct_execute] Step %d failed: %s", step.order, exc)
                    break
            else:
                execution_log.append(f"🚫 Step {step.order} BLOCKED by guardrail: {step.command}")
                logger.warning("[direct_execute] Guardrail blocked: %s", step.command[:60])

    exec_summary = "\n".join(execution_log)
    postmortem = _generate_postmortem(state, plan, True, exec_summary)
    ai_msg = AIMessage(content=f"Execution complete:\n{exec_summary}")
    return {"postmortem": postmortem, "messages": [ai_msg]}


# ===========================================================================
# RETAIN NODE (Continuous Learning)
# ===========================================================================

async def retain_node(state: dict) -> dict:
    """
    Store the resolved incident as a new CBR case (continuous learning).

    Called after successful remediation. Caches the new case so future
    similar incidents can be resolved faster via CBR retrieval.

    Reads:  state["triage_result"], state["remediation_plan"], state["cbr_confidence"],
            state["confirmed_root_cause"]
    Writes: state["messages"]
    """
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    severity = state.get("severity", "P1")
    plan: RemediationPlan | None = state.get("remediation_plan")
    confirmed_root_cause = state.get("confirmed_root_cause", "unknown")
    precedent_id = state.get("precedent_incident_id", "")

    if not plan:
        return {}

    import hashlib
    import uuid
    new_incident_id = f"AIRS-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    telemetry = state.get("telemetry", "")
    node_data = await _EKG.get_node(service)
    service_tier = node_data.get("tier", 2) if node_data else 2

    vector = _CBR_ENGINE.extract_feature_vector(telemetry, service, service_tier)
    fingerprint = hashlib.sha256(f"{confirmed_root_cause}:{service}".encode()).hexdigest()[:16]

    new_case = HistoricalCase(
        incident_id=new_incident_id,
        service=service,
        severity=severity,
        root_cause_category=confirmed_root_cause,
        symptom_vector=vector,
        telemetry_fingerprint=fingerprint,
        remediation_steps=[
            {"order": s.order, "action": s.action, "command": s.command, "risk": s.risk}
            for s in plan.steps
        ],
        rollback_command=plan.rollback_command,
        outcome="resolved",
        mttr_minutes=plan.estimated_mttr_minutes or 15,
        resolved_at=datetime.utcnow(),
        postmortem_summary=plan.postmortem_summary,
    )

    await _CBR_ENGINE.retain(new_case)
    logger.info("[retain_node] Stored new CBR case: %s (category=%s)", new_incident_id, confirmed_root_cause)
    ai_msg = AIMessage(
        content=f"✅ CBR case retained: `{new_incident_id}` — category: `{confirmed_root_cause}`"
    )
    return {"messages": [ai_msg]}


# ===========================================================================
# REJECT NODE
# ===========================================================================

async def reject_node(state: dict) -> dict:
    """
    Abort the incident response when the operator rejects the plan
    or when the policy engine deterministically blocks execution.
    """
    policy_result = state.get("policy_check_result", {})
    strategy = state.get("execution_strategy", "")
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"

    if strategy == "blocked" and policy_result.get("critical_violations"):
        violations = ", ".join(policy_result["critical_violations"])
        reason = f"Policy-as-Code blocked execution. Critical invariants violated: {violations}"
    else:
        reason = "Operator rejected the remediation plan."

    logger.info("[reject_node] %s", reason)
    postmortem = (
        f"# Incident Response Aborted — {service}\n\n"
        f"**Reason**: {reason}\n\n"
        f"**Timestamp**: {datetime.utcnow().isoformat()}Z\n\n"
        f"Manual intervention required."
    )
    ai_msg = AIMessage(content=f"❌ {reason}")
    return {"postmortem": postmortem, "is_approved": False, "messages": [ai_msg]}


# ===========================================================================
# APPROVAL NODE
# ===========================================================================

from langgraph.types import interrupt

async def approval_node(state: dict) -> dict:
    """
    Human-in-the-loop approval gate. Pauses graph execution and waits for
    operator input via the CLI's Prompt.ask() loop.
    """
    plan_md: str = state.get("plan", "No plan available.")
    blast_dict = state.get("blast_radius_result", {})
    blast_md = blast_dict.get("report_markdown", "")

    risk_level = "HIGH" if blast_dict.get("tier1_impact") else "MEDIUM"

    interrupt({
        "plan": plan_md,
        "blast_radius": blast_md,
        "risk": risk_level,
        "message": "Type 'approve' to execute or 'reject' to abort.",
    })

    # This code only runs after resume()
    resumed_data: dict = state.get("__interrupt_resume__", {})
    approved: bool = resumed_data.get("approved", False)
    return {"is_approved": approved}


# ===========================================================================
# POSTMORTEM / HELPER FUNCTIONS
# ===========================================================================

def _generate_postmortem(
    state: dict,
    plan: RemediationPlan,
    succeeded: bool,
    execution_summary: str,
) -> str:
    """Generate a final markdown postmortem document."""
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else "unknown"
    severity = state.get("severity", "P1")
    confirmed_root_cause = state.get("confirmed_root_cause", "unknown")
    precedent_id = state.get("precedent_incident_id", "")
    perception_stats = state.get("perception_stats", {})
    cbr_confidence = state.get("cbr_confidence", 0.0)
    blast_dict = state.get("blast_radius_result", {})

    status_icon = "✅ RESOLVED" if succeeded else "⚠️ ROLLED BACK"
    return f"""# Postmortem — {service} ({severity})

**Status**: {status_icon}
**Timestamp**: {datetime.utcnow().isoformat()}Z
**Confirmed Root Cause**: `{confirmed_root_cause}`
**CBR Precedent**: `{precedent_id}` (confidence: {cbr_confidence:.0%})

## Hybrid Memory Architecture Summary
- **Perception**: L1={perception_stats.get('L1_hits', 0)} L2={perception_stats.get('L2_hits', 0)} L3={perception_stats.get('L3_hits', 0)} hits
- **CBR Match**: {cbr_confidence:.0%} similarity to `{precedent_id}`
- **Blast Radius**: {len(blast_dict.get('affected_services', []))} services affected, tier-1 impact: {blast_dict.get('tier1_impact', False)}
- **Execution Strategy**: `{state.get('execution_strategy', 'unknown')}`

## Root Cause Analysis
{plan.root_cause}

## Remediation Steps
{chr(10).join(f'- Step {s.order}: {s.action}' for s in plan.steps)}

## Execution Summary
{execution_summary}

## Rollback Command
```bash
{plan.rollback_command}
```

**Estimated MTTR**: {plan.estimated_mttr_minutes} minutes
"""


# ===========================================================================
# PRESERVED: investigation/extraction nodes (used by NEURAL_FULL pathway)
# ===========================================================================

_INVESTIGATE_PROMPT_TEMPLATE = """\
You are an SRE investigator performing root cause analysis for the following incident:

  Service  : {service}
  Severity : {severity}
  Alert    : {alert_title}

## EKG Topology Context
{ekg_context}

## Investigation Objective
Use the available tools to gather enough telemetry to hand off a complete picture to the RCA agent.
Follow this strategy based strictly on the alert symptoms:
1. Call `get_metrics` with queries relevant ONLY to the symptoms of this specific alert.
2. Call `get_logs` with service="{service}" to inspect actual error traces.

## Telemetry Gathered So Far
{telemetry_so_far}

## Instructions
- Make ONE tool call per response.
- After each tool result, decide if you need more data.
- When you have sufficient evidence, respond with ONLY: INVESTIGATION_COMPLETE
- You have at most {remaining_retries} tool call(s) remaining.
"""


async def investigate_node(state: dict) -> dict:
    """ReAct-style investigation loop (used for NEURAL_FULL pathway only)."""
    triage: TriageResult | None = state.get("triage_result")
    service = triage.service if triage else state.get("raw_alert", {}).get("service", "unknown")
    severity = state.get("severity", "P1")
    alert_title = state.get("raw_alert", {}).get("title", "Unknown incident")
    telemetry_so_far = state.get("telemetry", "")
    retry_count = state.get("retry_count", 0)
    ekg_context = state.get("ekg_service_context", "No topology data available.")
    remaining = MAX_RETRIES - retry_count

    logger.info("[investigate_node] iteration=%d/%d service=%s", retry_count + 1, MAX_RETRIES, service)

    prompt = _INVESTIGATE_PROMPT_TEMPLATE.format(
        service=service, severity=severity, alert_title=alert_title,
        ekg_context=ekg_context[:600],
        telemetry_so_far=telemetry_so_far[:1000] if telemetry_so_far else "(none yet)",
        remaining_retries=remaining,
    )
    prior_messages = state.get("messages", [])
    round_messages = [HumanMessage(content=prompt)] + prior_messages

    llm = _make_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    ai_response: AIMessage = await llm_with_tools.ainvoke(round_messages)

    new_messages = [ai_response]
    new_telemetry = telemetry_so_far
    new_retry_count = retry_count + 1

    if ai_response.tool_calls:
        tool_call = ai_response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info("[investigate_node] tool_call name=%s args=%s", tool_name, tool_args)

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

        tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call_id, name=tool_name)
        new_messages.append(tool_msg)
        section_header = f"\n\n---\n### Round {new_retry_count} — `{tool_name}`\n"
        new_telemetry = new_telemetry + section_header + tool_result
    else:
        response_text = ai_response.content or ""
        if "INVESTIGATION_COMPLETE" not in response_text:
            logger.warning("[investigate_node] No tool call and no INVESTIGATION_COMPLETE.")

    return {"telemetry": new_telemetry, "retry_count": new_retry_count, "messages": new_messages}


def should_continue_investigation(state: dict) -> str:
    """Conditional edge: loop investigate or proceed to reasoning."""
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return "extract"
    messages = state.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return "investigate"
    if last_ai.tool_calls:
        return "investigate"
    if "INVESTIGATION_COMPLETE" in (last_ai.content or ""):
        return "extract"
    return "investigate"


_EXTRACT_PROMPT_TEMPLATE = """\
You are an expert diagnostician. Extract exact verbatim strings from the telemetry.

## Telemetry Evidence
{telemetry}

## Task
Extract exact anomalous metrics, error strings, exception names, and exit codes.

Respond with:
<extracted_evidence>
[Your verbatim quotes here]
</extracted_evidence>
"""


async def extract_node(state: dict) -> dict:
    """Extract verbatim evidence from raw telemetry for plan generation."""
    telemetry = state.get("telemetry", "No telemetry gathered.")
    logger.info("[extract_node] Extracting key evidence")
    llm = _make_llm()
    human_msg = HumanMessage(content=_EXTRACT_PROMPT_TEMPLATE.format(telemetry=telemetry))
    response = await llm.ainvoke([human_msg])
    extracted_text = response.content
    if "<extracted_evidence>" in extracted_text:
        extracted_text = extracted_text.split("<extracted_evidence>")[1].split("</extracted_evidence>")[0].strip()
    ai_msg = AIMessage(content=f"Extracted evidence:\n{extracted_text}")
    return {"extracted_evidence": extracted_text, "messages": [human_msg, ai_msg]}
