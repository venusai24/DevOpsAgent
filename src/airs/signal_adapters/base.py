"""
Signal Adapter Protocol — Module 1.6.

The SignalAdapter is the abstraction layer between the investigation
workflow and the raw MCP tool calls. Each signal type (METRICS, LOGS,
TRACES, K8S_STATE, CODE) has a dedicated adapter that:

1. execute()    — translates an ExecutionIntent to vendor-specific MCP call(s)
2. pre_filter() — deterministically reduces the raw response using math
                  (Z-score, dedup, error-span extraction) — NO LLM involved
3. normalize()  — converts vendor-specific filtered data to EvidenceCandidate

This three-stage pipeline ensures that LLM context is only populated with
pre-filtered, semantically meaningful signal, never raw noisy MCP responses.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from airs.models.evidence import EvidenceCandidate, EntityType, SignalSource
from airs.models.intents import ExecutionIntent, SignalType


@dataclass
class RawSignalPayload:
    """Raw, unfiltered MCP tool response."""
    server_id: str
    tool_name: str
    content: Any                # Parsed response (dict, list, or str)
    raw_text: str
    raw_digest: str
    idempotency_key: str
    arguments_hash: str
    success: bool
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    received_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.received_at is None:
            self.received_at = datetime.now(timezone.utc)


@dataclass
class FilteredSignalPayload:
    """Pre-filtered MCP response — ready for LLM interpretation."""
    server_id: str
    tool_name: str
    filtered_content: dict[str, Any]  # Structured, filtered data
    token_estimate: int                 # Rough token count of filtered_content
    filter_stats: dict[str, Any]        # Metadata about what was filtered
    raw_digest: str
    idempotency_key: str
    arguments_hash: str
    received_at: datetime


class SignalAdapter(ABC):
    """
    Abstract base for all signal type adapters.

    Each adapter implements execute() + pre_filter() + normalize() for one
    observability signal type. Adapters are stateless — all state is in the
    InvestigationState passed through the Temporal workflow.
    """

    @property
    @abstractmethod
    def signal_type(self) -> SignalType:
        """The signal type this adapter handles."""
        ...

    @abstractmethod
    async def execute(
        self,
        intent: ExecutionIntent,
        mcp_client: Any,  # MCPToolClient (avoid circular import)
    ) -> RawSignalPayload:
        """
        Translate ExecutionIntent → MCP tool call(s) → RawSignalPayload.
        May make multiple MCP calls and aggregate results.
        """
        ...

    @abstractmethod
    def pre_filter(
        self,
        payload: RawSignalPayload,
        incident_service: Optional[str] = None,
        incident_namespace: Optional[str] = None,
        time_window_minutes: int = 30,
    ) -> FilteredSignalPayload:
        """
        Deterministically reduce the raw MCP response using math.

        Must NOT use any LLM. Must be idempotent. Must discard irrelevant data.
        """
        ...

    @abstractmethod
    def normalize(
        self,
        filtered: FilteredSignalPayload,
        hop_index: int,
    ) -> EvidenceCandidate:
        """
        Convert a FilteredSignalPayload to an EvidenceCandidate.

        This is the final step before the ContextManager evaluates the
        candidate for inclusion in the Investigation Graph.
        """
        ...

    # ─── Shared helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(content: Any) -> int:
        """Rough token estimate: len(json_string) / 4."""
        import json
        try:
            text = json.dumps(content) if not isinstance(content, str) else content
            return max(1, len(text) // 4)
        except Exception:
            return 100
