"""State Management & Persistence subsystem for the DevOps Agent RCA framework.

Implements the complete four-store persistence architecture from:
  - StateManagement&PersistenceLayer.md (Phase 3)
  - RecoveryAndGuardrails.md (Phase 4)

Stores:
  1. Investigation Registry   — PostgreSQL (SERIALIZABLE dedup, GiST overlap queries)
  2. LangGraph Checkpointer   — PostgreSQL (full InvestigationState at every node transition)
  3. Stage 0 Artifact Cache   — Redis (version-keyed, TTL 6h)
  4. Baseline Statistics Store — PostgreSQL partitioned (per-component/KPI z-score data)

Public API surfaces:
  - ``state.models`` — all immutable state types
  - ``state.enums``  — all enumerations
  - ``state.interfaces`` — repository ABCs
  - ``state.serializers`` — full round-trip serialization
  - ``state.repositories`` — concrete repository implementations
  - ``state.adapters`` — infrastructure client wrappers
  - ``state.managers`` — CheckpointManager, ResumeManager, MigrationManager
  - ``state.migrations`` — idempotent SQL schema migrations
  - ``state.factories`` — RepositoryFactory (DI root), StateFactory
  - ``state.config`` — PersistenceConfig

Entry point for consumers::

    from devops_agent.state.factories import RepositoryFactory
    from devops_agent.state.config import PersistenceConfig

    config = PersistenceConfig.from_env()
    factory = RepositoryFactory(config)
    await factory.connect()

    # At application startup, apply migrations:
    await factory.migration_manager.migrate()
"""

from .config import PersistenceConfig
from .enums import (
    CircuitBreakerStatus,
    ConfidenceLevel,
    DedupDecision,
    FailureType,
    InvestigationStatus,
    StageOutcome,
)
from .factories import RepositoryFactory, StateFactory
from .models import (
    BaseState,
    CheckpointState,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
    ContextSnapshot,
    ExecutionMetrics,
    ExecutionState,
    ExplicitSymptoms,
    FailureEvent,
    FailureHistory,
    FallbackActivation,
    FallbackHistory,
    GuardrailState,
    InvestigationState,
    LoopGuardrailRecord,
    RecoveryAction,
    RecoveryState,
    RetryRecord,
    RetryState,
    StageMetrics,
    StageState,
    StageTimingRecord,
    StateMetadata,
    StateVersion,
    TimeoutState,
    ToolExecutionState,
    WorkflowState,
)

__all__ = [
    # Models
    "BaseState", "StateVersion", "StateMetadata",
    "ExecutionState", "WorkflowState", "StageState",
    "InvestigationState", "ExplicitSymptoms",
    "ToolExecutionState",
    "RetryState", "RetryRecord",
    "GuardrailState", "LoopGuardrailRecord", "CircuitBreakerSnapshot", "StageTimingRecord",
    "RecoveryState", "RecoveryAction",
    "CheckpointState",
    "TimeoutState",
    "CircuitBreakerState",
    "FailureHistory", "FailureEvent",
    "FallbackHistory", "FallbackActivation",
    "ExecutionMetrics", "StageMetrics",
    "ContextSnapshot",
    # Enums
    "InvestigationStatus", "DedupDecision", "FailureType",
    "CircuitBreakerStatus", "ConfidenceLevel", "StageOutcome",
    # Entry points
    "RepositoryFactory", "StateFactory", "PersistenceConfig",
]
