import json
from collections.abc import Sequence
from datetime import UTC
from typing import Any

from langchain_core.tools import StructuredTool

from devops_agent.tools.base import BaseTool
from devops_agent.tools.executor import ToolExecutor
from devops_agent.tools.models import ToolContext, ToolStatus


def _format_error(result) -> str:
    """Format error exactly as specified in ToolInterfaceDesign.md"""
    error_obj = {
        "error": getattr(result, "error_code", "UNKNOWN_ERROR") or "UNKNOWN_ERROR",
        "field": getattr(result, "error_field", "unknown"),
        "message": getattr(result, "error_message", "An unexpected error occurred.") or "An unexpected error occurred.",
        "hint": getattr(result, "error_hint", "Check input parameters and try again."),
        "partial_results": getattr(result, "partial_results", None)
    }
    return json.dumps(error_obj)

def create_langchain_tool(devops_tool: BaseTool, executor: ToolExecutor, ctx: ToolContext) -> StructuredTool:
    """Wraps a DevOpsAgent tool in a LangChain StructuredTool."""
    
    # Attach state to executor if not present
    if not hasattr(executor, "circuit_breakers"):
        executor.circuit_breakers = {}
    if not hasattr(executor, "dedup_cache"):
        executor.dedup_cache = {}
    if not hasattr(executor, "budget_spend"):
        executor.budget_spend = 0
        
    def sync_dummy(**kwargs):
        raise NotImplementedError("This tool is async only.")
        
    async def _arun(**kwargs: Any) -> Any:
        import hashlib
        from datetime import datetime

        from devops_agent.tools.models import ToolResult
        
        tool_name = devops_tool.metadata.name
        now = datetime.now(UTC)
        
        # 1. Budget Check (Q3)
        executor.budget_spend += 1
        if executor.budget_spend > 25:
            return _format_error(ToolResult(
                execution_id=ctx.execution_id, 
                status=ToolStatus.FAILED, 
                started_at=now,
                completed_at=now,
                error_code="BUDGET_EXCEEDED", 
                error_message="Investigation budget exceeded for this node."
            ))
            
        # 2. Circuit Breaker Check (Q4)
        cb_fails = executor.circuit_breakers.get(tool_name, 0)
        if cb_fails >= 3:
            return _format_error(ToolResult(
                execution_id=ctx.execution_id, 
                status=ToolStatus.FAILED, 
                started_at=now,
                completed_at=now,
                error_code="CIRCUIT_BREAKER_OPEN", 
                error_message=f"Tool {tool_name} failed 3 times consecutively. It is temporarily disabled."
            ))
        
        # 3. Fingerprint Dedup Check (Q2)
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        fingerprint = hashlib.md5(f"{tool_name}:{kwargs_str}".encode()).hexdigest()
        
        if fingerprint in executor.dedup_cache:
            return executor.dedup_cache[fingerprint]
        
        # 4 & 5: Timeout Wrapping & Post-call Result Recording are handled in ToolExecutor
        result = await executor.execute_tool(tool=devops_tool, raw_input=kwargs, ctx=ctx)
        
        if result.status == ToolStatus.SUCCESS:
            executor.circuit_breakers[tool_name] = 0 # Reset circuit breaker
            if hasattr(result.output, "model_dump"):
                out_str = json.dumps(result.output.model_dump())
            else:
                out_str = json.dumps(result.output)
            
            # Record in dedup cache
            executor.dedup_cache[fingerprint] = out_str
            return out_str
        else:
            executor.circuit_breakers[tool_name] = cb_fails + 1
            return _format_error(result)

    return StructuredTool.from_function(
        func=sync_dummy,
        coroutine=_arun,
        name=devops_tool.metadata.name,
        description=devops_tool.metadata.description,
        args_schema=devops_tool.schema.input_type,
    )

def wrap_tools(tools: Sequence[BaseTool], executor: ToolExecutor, ctx: ToolContext) -> list[StructuredTool]:
    """Wraps a list of DevOpsAgent tools into LangChain StructuredTools."""
    return [create_langchain_tool(t, executor, ctx) for t in tools]
