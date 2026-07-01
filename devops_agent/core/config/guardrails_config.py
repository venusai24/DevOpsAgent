"""Guardrails Configuration."""

from dataclasses import dataclass, field


@dataclass
class LLMRetryConfig:
    max_tool_failure_retries: int = 3
    max_schema_violation_retries: int = 2
    max_output_repair_attempts: int = 1
    base_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0
    on_exhaustion: str = "HITL_ESCALATION"

@dataclass
class LoopPreventionConfig:
    max_tool_calls_per_hypothesis: int = 12
    max_tool_calls_evidence_node: int = 60
    score_stagnation_threshold: float = 0.03
    score_stagnation_window: int = 4
    early_exit_score_floor: float = 0.10
    max_tool_calls_triage_node: int = 25
    max_tool_calls_context_node: int = 15

@dataclass
class TraceDegradationConfig:
    min_acceptable_coverage_pct: float = 0.60
    min_anomalous_trace_count: int = 3
    max_sampling_rate_estimate: float = 0.01
    bottleneck_concentration_min: float = 0.50
    metric_fallback_coverage_min: float = 0.60

@dataclass
class TimeoutConfig:
    global_timeout_seconds: int = 1800
    context_assembly_timeout_s: int = 300
    triage_timeout_s: int = 300
    evidence_collection_timeout_s: int = 600
    causal_reasoning_timeout_s: int = 300
    hitl_pause_watchdog_hours: int = 24
    report_delivery_timeout_s: int = 60

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    open_timeout_s: int = 120
    bypass_tools: list[str] = field(default_factory=lambda: [
        "run_connected_component_analysis",
        "run_propagation_direction_check",
    ])

@dataclass
class GuardrailsConfig:
    llm_retry: LLMRetryConfig = field(default_factory=LLMRetryConfig)
    loop_prevention: LoopPreventionConfig = field(default_factory=LoopPreventionConfig)
    trace_degradation: TraceDegradationConfig = field(default_factory=TraceDegradationConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
