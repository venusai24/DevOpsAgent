"""Exceptions for tool lifecycle and execution."""



class ToolError(Exception):
    """Base exception for all tool-related errors."""
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR"):
        self.error_code = error_code
        super().__init__(message)

class ToolValidationError(ToolError):
    """Raised when tool inputs fail schema or logic validation."""
    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}
        super().__init__(message, "VALIDATION_ERROR")

class ToolExecutionError(ToolError):
    """Raised when a tool encounters a runtime failure."""
    pass

class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds its allocated execution budget."""
    def __init__(self, message: str = "Tool execution timed out"):
        super().__init__(message, "TIMEOUT_ERROR")

class ToolCancellationError(ToolError):
    """Raised when tool execution is explicitly cancelled."""
    def __init__(self, message: str = "Tool execution cancelled"):
        super().__init__(message, "CANCELLED")

class ToolNotFoundError(ToolError):
    """Raised when looking up an unregistered tool."""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found in registry", "TOOL_NOT_FOUND")
