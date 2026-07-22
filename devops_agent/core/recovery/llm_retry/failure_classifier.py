"""LLM and Tool Failure Classification."""

from enum import StrEnum
from typing import Any

from pydantic import ValidationError


class FailureType(StrEnum):
    TOOL_ERROR_CORRECTABLE = "tool_error_correctable"
    TOOL_ERROR_UNCORRECTABLE = "tool_error_uncorrectable"
    TOOL_TIMEOUT = "tool_timeout"
    CIRCUIT_OPEN = "circuit_open"
    SCHEMA_VIOLATION = "schema_violation"
    FIELD_TYPE_ERROR = "field_type_error"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    HALLUCINATED_TOOL_CALL = "hallucinated_tool_call"
    UNPARSEABLE_JSON = "unparseable_json"
    LLM_REFUSAL = "llm_refusal"
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"

class FailureClassifier:
    CORRECTABLE_TOOL_CODES = {
        "UNKNOWN_CMDB_ID", "GENERIC_KPI_NAME", "INVALID_TIME_RANGE",
        "TOO_MANY_ITEMS", "INVALID_REGEX", "EMPTY_ANOMALY_INVENTORY",
    }
    UNCORRECTABLE_TOOL_CODES = {
        "CSV_NOT_FOUND", "INVALID_BASELINE_REF",
        "INSUFFICIENT_CLUSTER_SIZE", "INSUFFICIENT_T0_COVERAGE",
    }

    def classify_tool_result(self, tool_result: dict[str, Any]) -> FailureType | None:
        code = tool_result.get("error_code") or tool_result.get("error")
        if not code:
            return None
        if code == "TIMEOUT_ERROR":
            return FailureType.TOOL_TIMEOUT
        if code == "CIRCUIT_OPEN":
            return FailureType.CIRCUIT_OPEN
        if code in self.CORRECTABLE_TOOL_CODES:
            return FailureType.TOOL_ERROR_CORRECTABLE
        return FailureType.TOOL_ERROR_UNCORRECTABLE

    def classify_llm_output(self, output: Any, schema: type) -> FailureType | None:
        if not isinstance(output, (dict, str)):
            return FailureType.UNPARSEABLE_JSON
            
        if isinstance(output, str):
            # Assume we need to parse it (in a real system, json.loads would happen first)
            # If it's a string here, it failed JSON parsing upstream
            return FailureType.UNPARSEABLE_JSON
            
        try:
            schema.model_validate(output)
            return None
        except ValidationError as e:
            errors = e.errors()
            if any("missing" in str(err).lower() for err in errors):
                return FailureType.MISSING_REQUIRED_FIELD
            if any("type" in str(err).lower() for err in errors):
                return FailureType.FIELD_TYPE_ERROR
            return FailureType.SCHEMA_VIOLATION
