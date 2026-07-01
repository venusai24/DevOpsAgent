"""Telemetry, metrics, and tracing for tool execution."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import TypeVar

from langsmith import traceable

from .exceptions import ToolError
from .models import ToolResult, ToolStatus

logger = logging.getLogger("devops_agent.tools.telemetry")

T = TypeVar("T")
P = TypeVar("P")

@dataclass
class ToolTelemetry:
    """Records metrics for tool executions."""
    tool_name: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_elapsed_ms: float = 0.0

    def record_result(self, result: ToolResult) -> None:
        self.invocations += 1
        self.total_elapsed_ms += result.elapsed_ms
        if result.status == ToolStatus.SUCCESS:
            self.successes += 1
        elif result.status == ToolStatus.TIMEOUT:
            self.timeouts += 1
        else:
            self.failures += 1

class TelemetryManager:
    """In-memory telemetry sink (in production, exports to Prometheus/OTel)."""
    
    def __init__(self):
        self._metrics: dict[str, ToolTelemetry] = {}
        
    def get_telemetry(self, tool_name: str) -> ToolTelemetry:
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolTelemetry(tool_name)
        return self._metrics[tool_name]
        
    def record(self, tool_name: str, result: ToolResult) -> None:
        self.get_telemetry(tool_name).record_result(result)

_telemetry_sink = TelemetryManager()

def get_telemetry_manager() -> TelemetryManager:
    return _telemetry_sink

def with_telemetry(tool_name: str):
    """Decorator to automatically trace and record metrics for tool execution."""
    def decorator(func: Callable[..., Awaitable[ToolResult]]):
        @traceable(run_type="tool", name=tool_name)
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ToolResult:
            logger.debug("TRACING [%s]: started", tool_name)
            try:
                result = await func(*args, **kwargs)
                _telemetry_sink.record(tool_name, result)
                if result.status != ToolStatus.SUCCESS:
                    logger.warning("TRACING [%s]: %s - %s", tool_name, result.status.value, result.error_message)
                else:
                    logger.debug("TRACING [%s]: success (%.1fms)", tool_name, result.elapsed_ms)
                return result
            except ToolError as e:
                logger.error("TRACING [%s]: failed with %s: %s", tool_name, type(e).__name__, str(e))
                raise
            except Exception:
                logger.exception("TRACING [%s]: unexpected exception", tool_name)
                raise
        return wrapper
    return decorator
