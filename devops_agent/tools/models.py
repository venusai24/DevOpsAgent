"""Data models for the Tool Framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")

class ToolPermissions(StrEnum):
    READ_ONLY = "read_only"
    WRITE_BASELINE = "write_baseline"
    EXECUTE_EXTERNAL = "execute_external"

@dataclass
class ToolMetadata:
    """Static metadata for a registered tool."""
    name: str
    description: str
    version: str = "1.0.0"
    permissions: set[ToolPermissions] = field(default_factory=lambda: {ToolPermissions.READ_ONLY})
    author: str = "system"
    tags: set[str] = field(default_factory=set)

@dataclass
class ToolContext:
    """Context injected into every tool execution.
    Contains orchestration-agnostic contextual references like baseline_ref.
    """
    investigation_id: uuid.UUID
    cluster_id: str
    execution_id: uuid.UUID = field(default_factory=uuid.uuid4)
    baseline_ref: str | None = None
    app_stats_path: str | None = None
    metrics_path: str | None = None
    logs_path: str | None = None
    traces_path: str | None = None
    # Injected by the orchestrator node (Python code), never by the LLM.
    # The topology graph is a hard business artifact — graph traversal
    # logic must be deterministic code, not LLM-generated output.
    topology_graph: dict[str, list[str]] | None = field(default_factory=dict)
    timeout_seconds: int = 60
    attempt_number: int = 1
    # Dependency-injected clients will be passed separately via DI

@dataclass
class ToolSchema[T]:
    """Wrapper for Pydantic input schemas."""
    input_type: type[BaseModel]
    output_type: type[BaseModel]

class ToolStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class ToolResult[T]:
    """Structured output returned from a tool execution."""
    execution_id: uuid.UUID
    status: ToolStatus
    started_at: datetime
    completed_at: datetime
    output: T | None = None
    error_message: str | None = None
    error_code: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    
    @property
    def elapsed_ms(self) -> float:
        return (self.completed_at - self.started_at).total_seconds() * 1000.0

@dataclass
class ToolRetryPolicy:
    """Configurable retry policy per tool execution."""
    max_attempts: int = 3
    base_backoff_ms: int = 500
    max_backoff_ms: int = 5000
    retryable_error_codes: set[str] = field(default_factory=set)
