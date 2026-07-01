"""Activates Stage 5.1b."""

from datetime import UTC, datetime
from typing import Any

from .trace_quality_assessor import TraceQualityReport


class GracefulDegradationActivator:
    def activate(self, state: dict[str, Any], quality_report: TraceQualityReport) -> dict[str, Any]:
        """Returns state update dict applied before RCAAgent invocation."""
        now = datetime.now(UTC).isoformat()
        return {
            "guardrails": {
                "degradation_path_active": True,
                "degradation_triggered_at": now,
                "degradation_trigger_reasons": quality_report.trigger_reasons,
            },
            "investigation_gaps": state.get("investigation_gaps", []) + [{
                "type": "trace_degradation_activated",
                "reasons": quality_report.trigger_reasons,
                "coverage_pct": quality_report.coverage_pct,
                "action": "Metric-based topology inference (Stage 5.1b) activated",
            }],
        }

    def build_degradation_context_injection(self, quality_report: TraceQualityReport) -> str:
        reasons = "; ".join(quality_report.trigger_reasons)
        return (
            f"TRACE EVIDENCE IS INSUFFICIENT. Reasons: {reasons}. "
            f"You MUST use the metric-based fallback path (Stage 5.1b): "
            f"1. Call infer_metric_dependency_edges first to build topology. "
            f"2. If inference_quality is 'poor', call compute_metric_latency_correlation. "
            f"3. All edges will have edge_tag 'metric_inferred'. "
            f"4. Maximum confidence_level achievable is MEDIUM. "
            f"DO NOT call build_span_tree_summary — trace data is insufficient."
        )
