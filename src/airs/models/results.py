"""
Result Models.

Outputs from Temporal activities: tool execution results, epistemic
verification outcomes, and the operational state machine states.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from airs.models.edges import DirectedEdge
from airs.models.evidence import EvidenceNode
from airs.models.risk import StepRiskEntry


# ─── Operational State Machine ────────────────────────────────────────────────

class OpState(str, Enum):
    """
    The four valid operational states of the investigation agent.

    CONTINUE:      Proceed to next investigation hop.
    DIAGNOSE:      Sufficient evidence gathered — produce final report.
    ABSTAIN_PRUNE: Tool failed or uncertainty too high — backtrack/try other.
    ESCALATE:      Human intervention required (M_t >= 1/delta or undecidable).
    """
    CONTINUE = "CONTINUE"
    DIAGNOSE = "DIAGNOSE"
    ABSTAIN_PRUNE = "ABSTAIN_PRUNE"
    ESCALATE = "ESCALATE"


# ─── Tool Execution Result ────────────────────────────────────────────────────

class ToolExecutionResult(BaseModel):
    """
    Raw output from execute_mcp_tool_activity, after pre-filtering.

    success:              Whether the MCP call succeeded.
    raw_response_digest:  SHA-256 of the original raw response (for provenance).
    filtered_content:     Pre-filtered, normalised content from SignalAdapter.
    token_count:          Estimated token cost of filtered_content.
    mcp_server_id:        Which MCP server responded.
    tool_name:            Which tool was invoked.
    idempotency_key:      Redis key used/recorded for this call.
    error_code:           HTTP/gRPC error code if success == False.
    error_message:        Human-readable error if success == False.
    """
    success: bool
    raw_response_digest: str
    filtered_content: dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(default=0, ge=0)
    mcp_server_id: str
    tool_name: str
    idempotency_key: str
    error_code: Optional[int] = None
    error_message: Optional[str] = None


# ─── Playbook Retrieval Result ────────────────────────────────────────────────

class PlaybookResult(BaseModel):
    """Output from query_conann_playbook_activity."""
    collection: str
    documents: list[dict[str, Any]] = Field(default_factory=list)
    coverage_met: bool = False
    query_text: str = ""
    token_count: int = Field(default=0, ge=0)


# ─── Verification Result ──────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    """
    Output from verify_epistemic_quality_activity.

    This is the per-hop result of the full risk pipeline. It determines the
    operational state transition the workflow will apply.

    operational_state:     The state the agent should transition to.
    new_evidence_node:     The EvidenceNode to add to the graph (if CONTINUE).
    new_edges:             Graph edges to add.
    step_risk:             Full risk record for this hop.
    updated_martingale:    New M_t value after this hop.
    updated_missing_mass:  New M_miss value after this hop.
    abstain_reason:        Explanation if state == ABSTAIN_PRUNE.
    escalation_reason:     Explanation if state == ESCALATE.
    """
    operational_state: OpState
    new_evidence_node: Optional[EvidenceNode] = None
    new_edges: list[DirectedEdge] = Field(default_factory=list)
    step_risk: Optional[StepRiskEntry] = None
    updated_martingale: float = Field(default=1.0, ge=0.0)
    updated_missing_mass: float = Field(default=1.0, ge=0.0, le=1.0)
    abstain_reason: Optional[str] = None
    escalation_reason: Optional[str] = None
