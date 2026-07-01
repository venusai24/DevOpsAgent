"""LLM Output Repair Prompts and Recovery Strategies."""

from typing import Any

from .failure_classifier import FailureType


class OutputRepairer:
    """Builds localized repair prompts for the LLM to fix structured formatting."""
    
    def build_schema_repair_prompt(self, raw_output: dict[str, Any], schema: type, failure: FailureType) -> str:
        schema_json = schema.model_json_schema()
        return (
            f"ROLE\n--------\nYou are a strict output formatting repairer.\n\n"
            f"CONTEXT\n--------\nYour previous output failed schema validation. Error type: {failure.value}.\n"
            f"Expected Schema: {schema_json}\n\n"
            f"CONSTRAINTS & RULES\n--------\n"
            f"1. You MUST REFORMAT your previous response to match this exact schema.\n"
            f"2. DO NOT generate new reasoning, facts, or assumptions. Use ONLY the data from your previous response.\n"
            f"3. Output ONLY valid JSON matching the schema, with no markdown, conversational text, or explanation."
        )

    def build_json_repair_prompt(self) -> str:
        return (
            "ROLE\n--------\nYou are a strict output formatting repairer.\n\n"
            "CONTEXT\n--------\nYour previous output was not valid JSON.\n\n"
            "CONSTRAINTS & RULES\n--------\n"
            "1. You MUST return ONLY a valid JSON object matching the required schema.\n"
            "2. DO NOT include markdown blocks, conversational text, or trailing characters.\n"
            "3. DO NOT change the underlying reasoning or invent new data."
        )

class BaselineRefRecoveryStrategy:
    """Special recovery logic for INVALID_BASELINE_REF."""
    
    def attempt_recovery(self, state: dict[str, Any], orchestrator_context: Any = None) -> bool:
        """
        Attempts to recompute the baseline.
        Returns True if successfully recovered, False if it still fails.
        """
        # In a real system, this would call the compute_baseline_statistics tool directly.
        # For our architecture skeleton, we just return False to simulate hitting the HITL path 
        # if the baseline is truly invalid and unrecoverable in memory.
        return False
