"""Retry Coordinator for LLM nodes."""

import logging
import time
from collections.abc import Callable
from typing import Any

from ...config.guardrails_config import GuardrailsConfig
from .failure_classifier import FailureClassifier, FailureType
from .output_repairer import BaselineRefRecoveryStrategy, OutputRepairer

logger = logging.getLogger(__name__)

class ToolCallException(Exception):
    def __init__(self, tool_name: str, tool_result: dict[str, Any]):
        self.tool_name = tool_name
        self.tool_result = tool_result
        super().__init__(f"Tool {tool_name} failed: {tool_result.get('error')}")

class RetryCoordinator:
    """Wraps agent node invocations."""
    
    def __init__(self, config: GuardrailsConfig):
        self.config = config.llm_retry
        self.classifier = FailureClassifier()
        self.repairer = OutputRepairer()
        self.baseline_recovery = BaselineRefRecoveryStrategy()

    def execute_with_retry(
        self, agent_fn: Callable, state: dict[str, Any], node_name: str, output_schema: type
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        retry_log = []
        tool_attempt = schema_attempt = repair_attempt = 0

        while True:
            try:
                # 1. Execute agent logic
                result = agent_fn(state)
                
                # 2. Check schema
                failure = self.classifier.classify_llm_output(result, output_schema)
                if failure is None:
                    return result, retry_log

                # 3. Schema repair path
                if failure in (FailureType.SCHEMA_VIOLATION, FailureType.MISSING_REQUIRED_FIELD, FailureType.FIELD_TYPE_ERROR):
                    if schema_attempt >= self.config.max_schema_violation_retries:
                        return self._handle_uncorrectable(state, node_name, failure, retry_log)
                    
                    repair_prompt = self.repairer.build_schema_repair_prompt(result, output_schema, failure)
                    state = self._inject_correction(state, repair_prompt)
                    
                    retry_log.append({
                        "node_name": node_name, "failure_type": "schema_violation",
                        "attempt_number": schema_attempt + 1, "repaired": True, "resolution": "success"
                    })
                    schema_attempt += 1
                    continue

                # 4. JSON repair path
                if failure == FailureType.UNPARSEABLE_JSON:
                    if repair_attempt >= self.config.max_output_repair_attempts:
                        return self._handle_uncorrectable(state, node_name, failure, retry_log)
                    
                    state = self._inject_correction(state, self.repairer.build_json_repair_prompt())
                    repair_attempt += 1
                    continue

                # Unhandled failure type
                return self._handle_uncorrectable(state, node_name, failure, retry_log)

            except ToolCallException as exc:
                failure = self.classifier.classify_tool_result(exc.tool_result)

                if failure == FailureType.TOOL_ERROR_CORRECTABLE:
                    if tool_attempt >= self.config.max_tool_failure_retries:
                        return self._handle_uncorrectable(state, node_name, failure, retry_log)
                        
                    hint = exc.tool_result.get("hint", "Retry with corrected parameters.")
                    state = self._inject_tool_correction(state, exc.tool_name, hint)
                    
                    backoff = min(
                        self.config.base_backoff_seconds * (self.config.backoff_multiplier ** tool_attempt),
                        self.config.max_backoff_seconds
                    )
                    logger.info("Retrying correctable tool failure. Backoff: %.1fs", backoff)
                    time.sleep(backoff)
                    tool_attempt += 1
                    continue
                    
                elif failure == FailureType.TOOL_ERROR_UNCORRECTABLE:
                    if exc.tool_result.get("error_code") == "INVALID_BASELINE_REF":
                        if self.baseline_recovery.attempt_recovery(state):
                            continue # Successfully recovered baseline
                    return self._handle_uncorrectable(state, node_name, failure, retry_log)
                    
                else:
                    return self._handle_uncorrectable(state, node_name, failure, retry_log)

    def _inject_correction(self, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Injects a reasoning correction directly into the state."""
        state["_injected_correction"] = prompt
        return state
        
    def _inject_tool_correction(self, state: dict[str, Any], tool: str, hint: str) -> dict[str, Any]:
        return self._inject_correction(state, f"Tool {tool} failed. Hint: {hint}")

    def _handle_uncorrectable(self, state: dict[str, Any], node_name: str, failure_type: FailureType, retry_log: list[dict[str, Any]]) -> tuple[dict[str, Any], list]:
        logger.warning("Uncorrectable failure %s in node %s", failure_type, node_name)
        guardrails = state.get("guardrails", {})
        guardrails.update({
            "retry_log": retry_log,
            "forced_exit_triggered": True,
            "forced_exit_reason": f"Uncorrectable failure in {node_name}: {failure_type.value}",
        })
        
        # State update dict routed to hitl_agent_failure_node
        return {
            "investigation_state": "AGENT_FAILURE",
            "guardrails": guardrails
        }, retry_log
