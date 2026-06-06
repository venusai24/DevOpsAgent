import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ObservabilityTracer:
    """
    Chain of Thought logging for the agent.
    Writes a structured JSON trace to AGENT_LOGS_DIR/trace.json.
    """

    _instance: Optional['ObservabilityTracer'] = None

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or os.environ.get("AGENT_LOGS_DIR", ".")
        self.trace_file = os.path.join(self.log_dir, "trace.json")
        self.trace_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "phase": "Initialization",
            "metrics_analyzed": [],
            "hypotheses_generated": [],
            "actions_taken": [],
            "final_conclusion": None,
            "conclusion_time": None,
        }

    @classmethod
    def get_instance(cls) -> 'ObservabilityTracer':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing or multiple runs in one process)."""
        cls._instance = None

    def _flush(self) -> None:
        """Write the current state to disk."""
        try:
            # Ensure the directory exists
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.trace_file, "w", encoding="utf-8") as f:
                json.dump(self.trace_data, f, indent=2)
        except Exception as e:
            logger.error(f"[Tracer] Failed to write trace file: {e}")

    def update_phase(self, phase: str) -> None:
        """Update the current operating phase of the agent."""
        self.trace_data["phase"] = phase
        self.trace_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self._flush()

    def record_metrics_analyzed(self, metrics: List[str]) -> None:
        """Record which metrics/logs were looked at."""
        self.trace_data["metrics_analyzed"].extend(metrics)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for m in self.trace_data["metrics_analyzed"]:
            if m not in seen:
                deduped.append(m)
                seen.add(m)
        self.trace_data["metrics_analyzed"] = deduped
        self._flush()

    def record_hypotheses_generated(self, hypotheses: List[str]) -> None:
        """Record hypotheses generated during the reasoning phase."""
        self.trace_data["hypotheses_generated"].extend(hypotheses)
        seen = set()
        deduped = []
        for h in self.trace_data["hypotheses_generated"]:
            if h not in seen:
                deduped.append(h)
                seen.add(h)
        self.trace_data["hypotheses_generated"] = deduped
        self._flush()

    def record_action_taken(self, action_name: str, result: str) -> None:
        """Record an action taken by the agent."""
        self.trace_data["actions_taken"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action_name,
            "result": result
        })
        self._flush()

    def conclude_diagnosis(self, final_conclusion: str) -> None:
        """Record the final conclusion of the diagnosis."""
        self.trace_data["phase"] = "Conclusion"
        self.trace_data["final_conclusion"] = final_conclusion
        self.trace_data["conclusion_time"] = datetime.utcnow().isoformat() + "Z"
        self._flush()
