"""
airs_v2.action — Stage 4 Safe Execution & Policy Engine

Public API
----------
ExecutionEngine     — Policy-gated remediation orchestrator.
PolicyEnvelope      — Deterministic five-rule policy gate.
MockSlackGateway    — Mock Slack webhook for human-approval routing.
HealthMonitor       — Post-execution health oracle.

Types
-----
RemediationAction   — Declarative single-step action (never directly executed).
RemediationPlan     — Ordered list of RemediationAction items.
PolicyDecision      — Per-action policy evaluation result.
PolicyViolation     — Machine-readable rule violation.
HealthCheckResult   — Post-execution probe outcome for one service.
ExecutionResult     — Full audit record for one plan execution run.

Policy Rule IDs
---------------
PE_R1  NAMESPACE_ALLOWLIST      — target_namespace must be in allowlist.
PE_R2  BLAST_RADIUS_CAP         — blast radius must not exceed caps.
PE_R3  BUSINESS_HOURS           — restricted kinds blocked outside business hours.
PE_R4  RISK_ROUTING             — HIGH→PENDING_HUMAN; CRITICAL→REJECTED.
PE_R5  EXEC_SCRIPT_ALWAYS_HIGH  — exec_script always treated as HIGH risk.
"""

from airs_v2.action.types import (  # noqa: F401
    ActionKind,
    DecisionOutcome,
    ExecutionResult,
    HealthCheckResult,
    PolicyDecision,
    PolicyViolation,
    RemediationAction,
    RemediationPlan,
    RiskLevel,
)
from airs_v2.action.policy_envelope import (  # noqa: F401
    PolicyEnvelope,
    PE_R1, PE_R2, PE_R3, PE_R4, PE_R5,
    PE_R1_NAME, PE_R2_NAME, PE_R3_NAME, PE_R4_NAME, PE_R5_NAME,
    DEFAULT_ALLOWED_NAMESPACES,
    MAX_BLAST_RADIUS,
    MAX_CRITICAL_BLAST_RADIUS,
)
from airs_v2.action.slack_gateway import MockSlackGateway  # noqa: F401
from airs_v2.action.health_monitor import HealthMonitor  # noqa: F401
from airs_v2.action.executor import ExecutionEngine  # noqa: F401
