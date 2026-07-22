"""Circuit Breaker FSM logic."""

import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from ...config.guardrails_config import CircuitBreakerConfig

logger = logging.getLogger(__name__)

class CircuitBreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitOpenException(Exception):
    def __init__(self, tool_name: str):
        super().__init__(f"Circuit breaker for tool {tool_name} is OPEN")
        self.tool_result = {"error": "CIRCUIT_OPEN", "error_code": "CIRCUIT_OPEN"}

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.open_since: float = 0.0

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        self._check_state()
        
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitOpenException(self.name)
            
        try:
            result = fn(*args, **kwargs)
            # Treat dict responses with 'error' keys as failures
            if isinstance(result, dict) and result.get("error"):
                self._record_failure()
            else:
                self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _check_state(self):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.open_since >= self.config.open_timeout_s:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker %s transitioned to HALF_OPEN", self.name)

    def _record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker %s transitioned to CLOSED", self.name)
        self.failure_count = 0

    def _record_failure(self):
        self.failure_count += 1
        if self.state == CircuitBreakerState.HALF_OPEN or self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.open_since = time.time()
            logger.warning("Circuit breaker %s transitioned to OPEN", self.name)
