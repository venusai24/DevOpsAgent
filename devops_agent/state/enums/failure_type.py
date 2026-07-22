"""Failure type enumeration used by Domain 1 recovery (RecoveryAndGuardrails.md §4.1)."""

from enum import StrEnum


class FailureType(StrEnum):
    """All failure classes the FailureClassifier can produce.

    Correctable tool failures have a ``hint`` field that closes the retry loop.
    Uncorrectable failures route directly to HITL or a special recovery strategy.
    Reasoning failures target only the LLM's output formatting, not its reasoning.
    """

    # ── Tool-level failures (LLM call was structurally correct) ──────────────
    # Tool returned an error code that has a hint-based correction path.
    TOOL_ERROR_CORRECTABLE = "tool_error_correctable"

    # Tool returned an error code with no LLM-actionable fix.
    TOOL_ERROR_UNCORRECTABLE = "tool_error_uncorrectable"

    # Tool call exceeded the 30-second per-call ceiling.
    TOOL_TIMEOUT = "tool_timeout"

    # CircuitBreakerRegistry returned OPEN for this tool.
    CIRCUIT_OPEN = "circuit_open"

    # ── Reasoning / output failures (LLM response was malformed) ─────────────
    # Output failed Pydantic schema validation.
    SCHEMA_VIOLATION = "schema_violation"

    # Output has a field with the wrong Python type.
    FIELD_TYPE_ERROR = "field_type_error"

    # Output is missing one or more required fields.
    MISSING_REQUIRED_FIELD = "missing_required_field"

    # LLM called a tool name that does not exist in the registry.
    HALLUCINATED_TOOL_CALL = "hallucinated_tool_call"

    # Raw LLM output is not valid JSON.
    UNPARSEABLE_JSON = "unparseable_json"

    # ── Unrecoverable states ──────────────────────────────────────────────────
    # LLM refused to produce a response.
    LLM_REFUSAL = "llm_refusal"

    # LLM response exceeded the model's output token limit.
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"

    @property
    def is_retriable(self) -> bool:
        """Return True if a retry attempt is appropriate for this failure type."""
        return self in (
            FailureType.TOOL_ERROR_CORRECTABLE,
            FailureType.SCHEMA_VIOLATION,
            FailureType.FIELD_TYPE_ERROR,
            FailureType.MISSING_REQUIRED_FIELD,
            FailureType.UNPARSEABLE_JSON,
            FailureType.TOOL_TIMEOUT,
        )

    @property
    def is_tool_level(self) -> bool:
        """Return True when the LLM's call was structurally valid but the tool erred."""
        return self in (
            FailureType.TOOL_ERROR_CORRECTABLE,
            FailureType.TOOL_ERROR_UNCORRECTABLE,
            FailureType.TOOL_TIMEOUT,
            FailureType.CIRCUIT_OPEN,
        )
