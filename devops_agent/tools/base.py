"""Base class for all tools in the framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import ToolExecutionError, ToolValidationError
from .models import ToolContext, ToolMetadata, ToolResult, ToolSchema, ToolStatus

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

class BaseTool[I: BaseModel, O: BaseModel](ABC):
    """Abstract base class for all Agentic RCA tools."""

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return static metadata for this tool."""

    @property
    @abstractmethod
    def schema(self) -> ToolSchema[I]:
        """Return the Pydantic schema for input and output."""

    def validate_input(self, raw_input: dict[str, Any]) -> I:
        """Validate raw dictionary input against the Pydantic schema."""
        try:
            return self.schema.input_type.model_validate(raw_input)
        except ValidationError as e:
            raise ToolValidationError(
                message=f"Input validation failed for {self.metadata.name}",
                details=e.errors()
            ) from e

    async def _execute_internal(self, ctx: ToolContext, validated_input: I) -> ToolResult[O]:
        """Wrap the abstract execute method with lifecycle tracking."""
        from datetime import datetime
        started_at = datetime.now(UTC)
        
        try:
            output = await self.execute(ctx, validated_input)
            
            # Allow tools to return raw dictionaries matching the output schema
            if isinstance(output, dict):
                output = self.schema.output_type.model_validate(output)
            elif not isinstance(output, self.schema.output_type):
                raise ToolExecutionError(
                    f"Tool returned {type(output).__name__}, expected {self.schema.output_type.__name__}"
                )
                
            return ToolResult(
                execution_id=ctx.execution_id,
                status=ToolStatus.SUCCESS,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                output=output
            )
        except Exception as e:
            # Let the executor handle retries and timeouts, we just package the failure
            error_code = getattr(e, 'error_code', 'INTERNAL_ERROR')
            return ToolResult(
                execution_id=ctx.execution_id,
                status=ToolStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                error_message=str(e),
                error_code=error_code
            )

    @abstractmethod
    async def execute(self, ctx: ToolContext, inputs: I) -> O | dict[str, Any]:
        """The core business logic of the tool. Must be implemented by subclasses."""

    async def stream(self, ctx: ToolContext, inputs: I) -> AsyncIterator[Any]:
        """Optional streaming execution (e.g. for log tailing). 
        Default implementation yields the final output once.
        """
        result = await self.execute(ctx, inputs)
        yield result
