"""Executor for running tools with timeouts, retries, and isolation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from .base import BaseTool
from .exceptions import ToolCancellationError
from .models import ToolContext, ToolResult, ToolRetryPolicy, ToolStatus
from .telemetry import with_telemetry

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Handles execution isolation, timeouts, and retries for tools."""
    
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry
        self._execution_history: dict[uuid.UUID, ToolResult] = {}
        
    def get_history(self) -> Sequence[ToolResult]:
        return list(self._execution_history.values())

    @with_telemetry("ToolExecutor.execute_tool")
    async def execute_tool(
        self,
        tool: BaseTool,
        raw_input: dict[str, Any],
        ctx: ToolContext,
        retry_policy: ToolRetryPolicy | None = None
    ) -> ToolResult:
        """Execute a tool with full lifecycle management."""
        retry_policy = retry_policy or ToolRetryPolicy()
        
        # 1. Validation (Synchronous)
        try:
            validated_input = tool.validate_input(raw_input)
        except Exception as e:
            res = ToolResult(
                execution_id=ctx.execution_id,
                status=ToolStatus.FAILED,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error_message=str(e),
                error_code="VALIDATION_ERROR"
            )
            self._execution_history[ctx.execution_id] = res
            return res

        # 2. Execution with Retries and Timeout
        for attempt in range(1, retry_policy.max_attempts + 1):
            ctx.attempt_number = attempt
            
            try:
                # Wrap execution in asyncio.wait_for
                # This ensures the tool cannot exceed its allocated budget
                result = await asyncio.wait_for(
                    tool._execute_internal(ctx, validated_input),
                    timeout=ctx.timeout_seconds
                )
                
                # Check for success or non-retryable failure
                if result.status == ToolStatus.SUCCESS:
                    self._execution_history[ctx.execution_id] = result
                    return result
                
                # Non-retryable error: return immediately without further attempts
                if result.error_code not in retry_policy.retryable_error_codes:
                    self._execution_history[ctx.execution_id] = result
                    return result

                # Retryable error on the final attempt: give up
                if attempt == retry_policy.max_attempts:
                    self._execution_history[ctx.execution_id] = result
                    return result
                    
            except TimeoutError:
                logger.warning("Tool %s timed out after %ds", tool.metadata.name, ctx.timeout_seconds)
                result = ToolResult(
                    execution_id=ctx.execution_id,
                    status=ToolStatus.TIMEOUT,
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    error_message=f"Execution exceeded {ctx.timeout_seconds}s timeout",
                    error_code="TIMEOUT_ERROR"
                )
                if attempt == retry_policy.max_attempts:
                    self._execution_history[ctx.execution_id] = result
                    return result
                    
            except asyncio.CancelledError:
                logger.info("Tool %s execution cancelled", tool.metadata.name)
                result = ToolResult(
                    execution_id=ctx.execution_id,
                    status=ToolStatus.CANCELLED,
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    error_message="Execution cancelled by orchestrator",
                    error_code="CANCELLED"
                )
                self._execution_history[ctx.execution_id] = result
                raise ToolCancellationError()
            
            # Backoff before next attempt
            if attempt < retry_policy.max_attempts:
                backoff_s = min(
                    retry_policy.base_backoff_ms * (2 ** (attempt - 1)),
                    retry_policy.max_backoff_ms
                ) / 1000.0
                logger.info("Retrying %s (attempt %d/%d) in %.1fs", tool.metadata.name, attempt + 1, retry_policy.max_attempts, backoff_s)
                await asyncio.sleep(backoff_s)

        # Fallback if loop exits without returning
        self._execution_history[ctx.execution_id] = result
        return result

    async def execute_parallel(
        self,
        tasks: Sequence[tuple[BaseTool, dict[str, Any], ToolContext]],
        retry_policy: ToolRetryPolicy | None = None
    ) -> Sequence[ToolResult]:
        """Execute multiple tools concurrently."""
        coros = [
            self.execute_tool(tool, raw_input, ctx, retry_policy)
            for tool, raw_input, ctx in tasks
        ]
        return await asyncio.gather(*coros, return_exceptions=True)
