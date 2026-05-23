"""
agent/state.py

Defines the complete type contract for the AIRS LangGraph execution graph.

Three concerns are handled here:

1.  **GraphState** — The single canonical TypedDict that is threaded through
    every node in the LangGraph graph. All agent outputs are written into
    this dict; no node maintains internal state.

2.  **TriageResult** — Pydantic v2 model that captures the structured output
    of the Triage Agent. Used as the `response_format` schema when calling
    Gemini to guarantee a valid, parseable JSON envelope.

3.  **RemediationPlan** — Pydantic v2 model that captures the structured
    output of the RCA & Recommendation Agent. The `rollback_command` field
    triggers a hard-coded guardrail: plans without a rollback are flagged
    high-risk and cannot proceed to the approval node.
"""

from __future__ import annotations

from typing import Annotated, Literal, Any, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Pydantic Output Schemas
# ---------------------------------------------------------------------------


class TriageResult(BaseModel):
    """
    Structured output produced by the Triage Agent.

    The agent is instructed to populate this schema via Gemini structured
    output (``with_structured_output``).  Strict typing on ``severity``
    ensures the conditional routing logic downstream never receives an
    out-of-range value.
    """

    severity: Literal["P0", "P1", "P2", "P3"] = Field(
        ...,
        description=(
            "Incident priority level following PagerDuty convention. "
            "P0 = customer-facing outage requiring immediate escalation; "
            "P1 = major degradation; P2 = minor degradation; P3 = informational."
        ),
    )
    service: str = Field(
        ...,
        min_length=1,
        description=(
            "Canonical name of the affected microservice as it appears in "
            "the monitoring platform (e.g. 'payments-service')."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Model's self-reported confidence in the severity classification "
            "(0.0 – 1.0). Values below 0.6 will be logged as low-confidence "
            "and may trigger a re-triage loop."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "One-sentence rationale for the assigned severity level. "
            "Captured for LangSmith trace visibility and postmortem generation."
        ),
    )

    @field_validator("service")
    @classmethod
    def normalise_service_name(cls, v: str) -> str:
        """Strip accidental whitespace from LLM output."""
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "severity": "P0",
                    "service": "payments-service",
                    "confidence": 0.97,
                    "reasoning": (
                        "Connection pool is at 100% saturation with 37 queued "
                        "requests and a 47% HTTP error rate, indicating a "
                        "complete customer-facing outage."
                    ),
                }
            ]
        }
    }


class RemediationStep(BaseModel):
    """A single, ordered remediation action."""

    order: int = Field(..., ge=1, description="Execution order (1-indexed).")
    action: str = Field(..., min_length=1, description="Human-readable action description.")
    command: str | None = Field(
        default=None,
        description="Optional shell or kubectl command to execute this step.",
    )
    risk: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Risk level of this individual step.",
    )


class RemediationPlan(BaseModel):
    """
    Structured output produced by the RCA & Recommendation Agent.

    **Guardrail**: If ``rollback_command`` is an empty string, the property
    ``is_high_risk`` evaluates to ``True``. The orchestrator will refuse to
    route the plan to the approval node without operator override, implementing
    a zero-trust safety layer purely in deterministic Python code.
    """

    root_cause: str = Field(
        ...,
        min_length=10,
        description=(
            "A precise, single-paragraph root cause analysis synthesised from "
            "the raw telemetry. Must cite specific metric values or log lines."
        ),
    )
    steps: list[RemediationStep] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of remediation actions. Each step must be atomic "
            "and independently verifiable."
        ),
    )
    rollback_command: str = Field(
        default="",
        description=(
            "A single shell or kubectl command that fully reverts every change "
            "made by the remediation steps. An empty string marks this plan as "
            "high-risk and blocks automated execution."
        ),
    )
    estimated_mttr_minutes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Estimated mean-time-to-recover in minutes. Optional; used in the "
            "postmortem document."
        ),
    )
    postmortem_summary: str = Field(
        default="",
        description=(
            "Optional brief summary paragraph for the postmortem document, "
            "pre-populated by the agent for human review."
        ),
    )

    # ------------------------------------------------------------------
    # Computed properties (zero-trust guardrails)
    # ------------------------------------------------------------------

    @property
    def is_high_risk(self) -> bool:
        """
        Returns True when no rollback command has been provided.
        High-risk plans are blocked from automated execution and require
        explicit operator override in the CLI.
        """
        return not self.rollback_command.strip()

    @property
    def step_commands(self) -> list[str]:
        """Convenience accessor: ordered list of non-null step commands."""
        return [s.command for s in self.steps if s.command]

    @model_validator(mode="after")
    def warn_on_missing_rollback(self) -> "RemediationPlan":
        """Attach a high-risk flag annotation to the postmortem summary."""
        if self.is_high_risk and self.postmortem_summary:
            self.postmortem_summary = (
                "[HIGH-RISK: No rollback command supplied] " + self.postmortem_summary
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "root_cause": (
                        "The payments-service database connection pool reached its hard "
                        "limit of 100 connections at 14:15 UTC. Log analysis identified "
                        "transaction 'tx-8f3a9d' open for 312 seconds — a connection "
                        "leak introduced by a missing 'finally' block in the payment "
                        "processing handler deployed at 13:50 UTC (commit a3f7d9c)."
                    ),
                    "steps": [
                        {
                            "order": 1,
                            "action": "Kill the long-running leaked transaction.",
                            "command": "kubectl exec -n prod deployment/payments-service -- psql -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < now() - interval '5 minutes';\"",
                            "risk": "medium",
                        },
                        {
                            "order": 2,
                            "action": "Rolling restart the payments-service pods to drain the connection pool.",
                            "command": "kubectl rollout restart deployment/payments-service -n prod",
                            "risk": "medium",
                        },
                        {
                            "order": 3,
                            "action": "Roll back the deployment to the previous stable image.",
                            "command": "kubectl rollout undo deployment/payments-service -n prod",
                            "risk": "low",
                        },
                    ],
                    "rollback_command": "kubectl rollout undo deployment/payments-service -n prod",
                    "estimated_mttr_minutes": 12,
                    "postmortem_summary": (
                        "A connection leak caused by a missing finally block in commit "
                        "a3f7d9c exhausted the DB pool, producing a P0 outage for ~8 "
                        "minutes. Remediation involved killing the leaked transaction and "
                        "rolling back the deployment."
                    ),
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Phase 2-5: New models for Hybrid Memory Architecture
# ---------------------------------------------------------------------------


class CausalNode(BaseModel):
    """
    A node in the sparse symbolic causal graph built by the logic_agent_node.
    Represents a single service/component as a root cause candidate.
    """
    service: str = Field(..., description="Service or component name.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim log/metric citations that implicate this node.",
    )
    is_root_cause: bool = Field(
        default=False,
        description="True if symbolic inference confirms this as the root cause.",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in root cause attribution (0.0\u20131.0).",
    )
    pruned_reason: str = Field(
        default="",
        description="If pruned, the reason (e.g., 'CPU nominal \u2014 cannot be root cause').",
    )


class BlastRadiusResult(BaseModel):
    """Serializable blast radius report stored in GraphState."""
    target_service: str
    affected_services: list[str] = Field(default_factory=list)
    tier1_services: list[str] = Field(default_factory=list)
    tier1_impact: bool = False
    risk_score: float = 0.0
    recommendation: str = "require_approval"  # auto_execute | require_approval | block
    on_call_contacts: list[str] = Field(default_factory=list)
    estimated_user_impact_pct: float = 0.0
    report_markdown: str = ""


class CBRMatch(BaseModel):
    """A single Case-Based Reasoning match result stored in GraphState."""
    incident_id: str
    service: str
    root_cause_category: str
    similarity_score: float
    mttr_minutes: int
    outcome: str
    postmortem_summary: str


class PolicyResult(BaseModel):
    """Policy-as-Code check result stored in GraphState."""
    passed: bool
    critical_violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    terraform_valid: bool = True
    terraform_errors: list[str] = Field(default_factory=list)
    result_markdown: str = ""


class CanaryStatus(BaseModel):
    """Progressive canary deployment status stored in GraphState."""
    service: str
    succeeded: bool = False
    stages_completed: int = 0
    halted_at_stage: Optional[int] = None
    total_duration_seconds: float = 0.0
    final_error_rate_pct: float = 0.0
    final_latency_p99_ms: float = 0.0
    rollback_executed: bool = False
    report_markdown: str = ""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class GraphState(dict):
    """
    The canonical state schema for the AIRS LangGraph graph.

    Implemented as a TypedDict so LangGraph can introspect field types for
    checkpointing and streaming. All mutable state lives here; nodes receive
    the full state dict and return a partial dict of only the fields they
    modify.

    **Message accumulation**: The ``messages`` field uses ``add_messages``
    as its reducer so that each node's returned messages are *appended* to
    the existing list rather than replacing it, preserving the full
    conversation history across the graph's checkpoints.

    **Retry guard**: ``retry_count`` is incremented by the investigation node
    on each tool-call cycle. Conditional routing logic caps this at 3 to
    prevent runaway API spend in case the LLM gets stuck in a tool loop.
    """

    # ------------------------------------------------------------------ #
    #  Message history (LangChain / LangGraph convention)                 #
    # ------------------------------------------------------------------ #
    messages: Annotated[list[AnyMessage], add_messages]
    """
    Full conversation message history (HumanMessage, AIMessage, ToolMessage).
    Uses the add_messages reducer so nodes can append without knowing the
    current list length.
    """

    # ------------------------------------------------------------------ #
    #  Incident context (written once by the CLI bootstrap)               #
    # ------------------------------------------------------------------ #
    raw_alert: dict
    """
    The raw PagerDuty-style alert JSON that triggered this graph execution.
    Written once at graph entry; never mutated by subsequent nodes.
    """

    # ------------------------------------------------------------------ #
    #  Triage outputs                                                      #
    # ------------------------------------------------------------------ #
    severity: str
    """
    The P-level string produced by the Triage Agent (e.g. 'P0').
    Drives the first conditional routing decision.
    """

    triage_result: TriageResult | None
    """
    The full Pydantic TriageResult model instance for downstream nodes
    that need more than just the severity level string.
    """

    # ------------------------------------------------------------------ #
    #  Investigation outputs                                               #
    # ------------------------------------------------------------------ #
    telemetry: str
    """
    A concatenated markdown-formatted string of all telemetry gathered
    during the investigation phase. Built up incrementally across multiple
    tool-call iterations. Fed verbatim to the RCA Agent as context.
    """

    retry_count: int
    """
    Incremented once per investigation loop iteration.
    Capped at 3 by conditional routing to prevent infinite tool loops.
    """

    extracted_evidence: str
    """
    The XML scratchpad containing verbatim metrics and error codes extracted
    from the raw telemetry by the extract_node. Passed to the RCA plan_node.
    """

    # ------------------------------------------------------------------ #
    #  RCA & Remediation outputs                                           #
    # ------------------------------------------------------------------ #
    plan: str
    """
    The markdown-serialised representation of the RemediationPlan.
    Used by the CLI's Rich renderer and the approval node's interrupt payload.
    """

    remediation_plan: RemediationPlan | None
    """
    The full typed Pydantic RemediationPlan instance. Checked by the
    zero-trust guardrail before routing to the approval node.
    """

    # ------------------------------------------------------------------ #
    #  Approval / execution                                                #
    # ------------------------------------------------------------------ #
    is_approved: bool
    """
    Set to True by the Human Approval node after the operator types 'approve'
    at the CLI prompt. Only the execute_node reads this field.
    """

    postmortem: str
    """
    The markdown postmortem document generated by execute_node after
    successful remediation. Written to disk by the CLI as a .md file.
    """

    # ------------------------------------------------------------------ #
    #  Phase 1: Enterprise Knowledge Graph outputs                         #
    # ------------------------------------------------------------------ #
    topology_map: dict
    """
    Serialized SubGraph extracted by the topology_agent_node.
    Contains the dependency chain and health status of all services
    in the blast radius of the failing service.
    """

    ekg_service_context: str
    """
    Markdown-formatted EKG context summary injected into agent prompts.
    Pre-rendered by the topology_agent_node for LLM consumption.
    """

    # ------------------------------------------------------------------ #
    #  Phase 2: Case-Based Reasoning outputs                               #
    # ------------------------------------------------------------------ #
    cbr_matches: list
    """
    Ranked list of CBRMatch dicts from the diagnostic_agent_node.
    Populated by the CBR engine's retrieve() phase.
    """

    cbr_confidence: float
    """
    Cosine similarity score of the best CBR match (0.0\u20131.0).
    High confidence (\u22650.8) means the system uses the CBR plan directly.
    """

    precedent_incident_id: str
    """
    ID of the historical case used as the primary CBR precedent.
    Included in the postmortem for auditability.
    """

    # ------------------------------------------------------------------ #
    #  Phase 3: Perception Layer outputs                                   #
    # ------------------------------------------------------------------ #
    perception_stats: dict
    """
    L1/L2/L3 tier hit statistics from the TieredLogClassifier.
    Structure: {L1_hits: int, L2_hits: int, L3_hits: int, primary_template: str}
    """

    primary_log_template: str
    """
    The dominant log pattern category identified by the perception layer.
    e.g., 'connection_pool_exhausted', 'oom_killed', 'dns_resolution_failure'.
    Used by the logic_agent_node for symbolic hypothesis pruning.
    """

    # ------------------------------------------------------------------ #
    #  Phase 4: Multi-agent reasoning outputs                              #
    # ------------------------------------------------------------------ #
    causal_graph_nodes: list
    """
    Serialized list of CausalNode dicts from the logic_agent_node.
    Represents the sparse causal graph after symbolic pruning.
    """

    confirmed_root_cause: str
    """
    The deterministically confirmed root cause category after logic_agent
    symbolic pruning (e.g., 'connection_pool_exhaustion').
    Empty string if logic agent could not deterministically confirm.
    """

    root_cause_hypotheses: list
    """
    Ranked list of root cause hypothesis dicts from the diagnostic_agent_node.
    Each dict: {category: str, confidence: float, evidence: list[str]}
    """

    # ------------------------------------------------------------------ #
    #  Phase 5: Safe execution outputs                                     #
    # ------------------------------------------------------------------ #
    blast_radius_result: dict
    """
    Serialized BlastRadiusResult dict from the risk_agent_node.
    Determines whether execution requires human approval.
    """

    policy_check_result: dict
    """
    Serialized PolicyResult dict from the policy_check_node.
    If passed=False, the plan is deterministically blocked.
    """

    execution_strategy: str
    """
    Determined by the blast radius + policy check.
    One of: 'canary' | 'direct' | 'blocked' | 'require_approval'
    """

    canary_status: dict
    """
    Serialized CanaryStatus dict from the canary_execute_node.
    Tracks per-stage golden signal observations.
    """

    rollback_triggered: bool
    """
    True if the rollback controller triggered an automatic rollback
    during or after canary execution.
    """


# ---------------------------------------------------------------------------
# Default state factory
# ---------------------------------------------------------------------------


def make_initial_state(raw_alert: dict) -> dict:
    """
    Return a fully-initialised GraphState dictionary suitable for passing
    to ``graph.astream_events()`` or ``graph.ainvoke()`` as the first
    argument.

    All optional / accumulative fields are given safe defaults so that
    every node can read any field without KeyError.

    Args:
        raw_alert: The PagerDuty-style alert dict that triggered the run.

    Returns:
        A dict conforming to the GraphState schema with all fields populated.
    """
    return {
        "messages": [],
        "raw_alert": raw_alert,
        "severity": "",
        "triage_result": None,
        "telemetry": "",
        "retry_count": 0,
        "extracted_evidence": "",
        "plan": "",
        "remediation_plan": None,
        "is_approved": False,
        "postmortem": "",
        # Phase 1: EKG
        "topology_map": {},
        "ekg_service_context": "",
        # Phase 2: CBR
        "cbr_matches": [],
        "cbr_confidence": 0.0,
        "precedent_incident_id": "",
        # Phase 3: Perception
        "perception_stats": {},
        "primary_log_template": "",
        # Phase 4: Multi-agent reasoning
        "causal_graph_nodes": [],
        "confirmed_root_cause": "",
        "root_cause_hypotheses": [],
        # Phase 5: Safe execution
        "blast_radius_result": {},
        "policy_check_result": {},
        "execution_strategy": "require_approval",
        "canary_status": {},
        "rollback_triggered": False,
    }
