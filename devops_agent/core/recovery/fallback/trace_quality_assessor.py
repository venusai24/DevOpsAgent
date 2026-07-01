"""Algorithmic Stage 5 criteria assessor."""

from dataclasses import dataclass
from typing import Any

from ...config.guardrails_config import TraceDegradationConfig


@dataclass
class TraceQualityReport:
    trace_quality_score: float
    coverage_pct: float
    anomalous_trace_count: int
    bottleneck_identified: bool
    bottleneck_component: str | None
    degradation_triggered: bool
    trigger_reasons: list[str]

class TraceQualityAssessor:
    def __init__(self, config: TraceDegradationConfig):
        self.config = config

    def assess(self, investigation_cluster: list[str], anomalous_traces_result: dict[str, Any], span_tree_summaries: list[dict[str, Any]]) -> TraceQualityReport:
        failed_criteria = 0
        trigger_reasons = []

        # Criterion 1: Count
        trace_count = len(anomalous_traces_result.get("trace_ids", []))
        if trace_count < self.config.min_anomalous_trace_count:
            failed_criteria += 1
            trigger_reasons.append(f"Trace count ({trace_count}) < {self.config.min_anomalous_trace_count}")

        # Criterion 2: Coverage
        covered = set()
        for summary in span_tree_summaries:
            covered.update(summary.get("components_present", []))
        coverage_pct = len(covered.intersection(investigation_cluster)) / len(investigation_cluster) if investigation_cluster else 0.0
        if coverage_pct < self.config.min_acceptable_coverage_pct:
            failed_criteria += 1
            trigger_reasons.append(f"Coverage ({coverage_pct:.0%}) < {self.config.min_acceptable_coverage_pct:.0%}")

        # Criterion 3: Bottleneck
        bottleneck_found = False
        bottleneck_comp = None
        for summary in span_tree_summaries:
            for span in summary.get("bottleneck_spans", []):
                if span.get("duration_pct", 0) > self.config.bottleneck_concentration_min:
                    bottleneck_found = True
                    bottleneck_comp = span.get("cmdb_id")
                    break
        if not bottleneck_found:
            failed_criteria += 1
            trigger_reasons.append(f"No bottleneck span > {self.config.bottleneck_concentration_min:.0%}")

        # Criterion 4: Sampling rate
        est_rate = anomalous_traces_result.get("estimated_sampling_rate", 1.0)
        if est_rate < self.config.max_sampling_rate_estimate:
            failed_criteria += 1
            trigger_reasons.append(f"Estimated sampling rate ({est_rate:.2%}) < {self.config.max_sampling_rate_estimate:.2%}")

        degradation = failed_criteria > 0
        score = 1.0 - (failed_criteria / 4)

        return TraceQualityReport(
            trace_quality_score=score,
            coverage_pct=coverage_pct,
            anomalous_trace_count=trace_count,
            bottleneck_identified=bottleneck_found,
            bottleneck_component=bottleneck_comp,
            degradation_triggered=degradation,
            trigger_reasons=trigger_reasons
        )
