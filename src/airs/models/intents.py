"""
Intent Models.

ExecutionIntent is the output of the LangGraph reasoning engine for each hop.
It is a discriminated union: exactly one of (tool_spec, playbook_query) is set,
depending on the action chosen by the route_decision node.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────

class IntentAction(str, Enum):
    """The four possible actions the reasoning engine can emit."""
    EXECUTE_TOOL = "EXECUTE_TOOL"
    QUERY_PLAYBOOK = "QUERY_PLAYBOOK"
    DIAGNOSE = "DIAGNOSE"
    ESCALATE = "ESCALATE"


class SignalType(str, Enum):
    """Observability signal type — maps to a specific SignalAdapter."""
    METRICS = "METRICS"
    LOGS = "LOGS"
    TRACES = "TRACES"
    K8S_STATE = "K8S_STATE"
    CODE = "CODE"
    # Local CSV telemetry files exported by the GUI module.
    # Processed through the DuckDB CSV Intelligence Layer.
    CSV = "CSV"


class ToolTier(int, Enum):
    """
    Tool tier classification per Design Document Section 6.2.

    OBSERVATION (1):  Read-only, idempotent, always safe.
    INVESTIGATIVE (2): May trigger side effects (e.g., log parsing at scale).
    REMEDIATION (3):  State-mutating. Requires PASC verification before use.
    """
    OBSERVATION = 1
    INVESTIGATIVE = 2
    REMEDIATION = 3


# ─── Tool Specification ───────────────────────────────────────────────────────

class ToolSpec(BaseModel):
    """
    Specification for a single MCP tool invocation.

    tool_name:     The MCP tool identifier (e.g., 'execute_query').
    mcp_server_id: Which MCP server to call (e.g., 'prometheus').
    arguments:     Tool-specific arguments dict (already templated).
    tier:          Risk tier of this tool.
    signal_type:   Observability signal this tool produces.
    idempotent:    Whether repeated calls with the same args are safe.
    expected_latency_seconds: Hint for Temporal activity timeout.
    """
    tool_name: str
    mcp_server_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tier: ToolTier = ToolTier.OBSERVATION
    signal_type: SignalType = SignalType.METRICS
    idempotent: bool = True
    expected_latency_seconds: int = Field(default=10, ge=1)


class PlaybookQuery(BaseModel):
    """
    Query specification for the ConANN playbook retrieval activity.

    collection:  Which of the 6 Qdrant collections to search.
    query_text:  Natural language query for embedding.
    filters:     Metadata pre-filters (applied before vector search).
    k:           Number of results to retrieve.
    """
    collection: str
    query_text: str
    filters: dict[str, Any] = Field(default_factory=dict)
    k: int = Field(default=3, ge=1, le=10)


# ─── Execution Intent ─────────────────────────────────────────────────────────

class ExecutionIntent(BaseModel):
    """
    The single output of the LangGraph reasoning engine per hop.

    action:        What the agent intends to do next.
    tool_spec:     Set when action == EXECUTE_TOOL.
    playbook_query: Set when action == QUERY_PLAYBOOK.
    reasoning:     Brief explanation of why this action was chosen.
    missing_mass_at_decision: M_t value that led to this decision.
    """
    action: IntentAction
    tool_spec: Optional[ToolSpec] = None
    playbook_query: Optional[PlaybookQuery] = None
    reasoning: str = ""
    missing_mass_at_decision: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ExecutionIntent":
        if self.action == IntentAction.EXECUTE_TOOL and self.tool_spec is None:
            raise ValueError("tool_spec must be set when action == EXECUTE_TOOL")
        if self.action == IntentAction.QUERY_PLAYBOOK and self.playbook_query is None:
            raise ValueError(
                "playbook_query must be set when action == QUERY_PLAYBOOK"
            )
        return self
