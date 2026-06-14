"""
airs_v2/action/feedback.py
============================

Structured HITL feedback parser (Stage 5).

Purpose
-------
Translates raw interaction payloads from three sources into the
``HumanFeedback`` Pydantic model:

  Slack : Block Kit modal submission payloads (future ProductionSlackGateway)
  CLI   : Structured JSON from stdin (for local development / testing)
  Mock  : Auto-generated feedback with configurable defaults (unit tests)

This module has no network calls or I/O side effects — it is a pure
data transformation layer. The gateway owns the transport; this module
owns the data contract.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Translates raw HITL interaction payloads into structured HumanFeedback.

    Parameters
    ----------
    prompt_sent_at : ISO-8601 timestamp when the HITL prompt was dispatched.
                     Used to compute time_to_feedback_s.
    """

    def __init__(self, prompt_sent_at: str = "") -> None:
        self._prompt_sent_at = prompt_sent_at

    # ── Slack payload parser ───────────────────────────────────────────────────

    def parse_slack_payload(self, payload: dict[str, Any]) -> object:
        """
        Parse a Slack Block Kit modal submission payload into HumanFeedback.

        Expected payload shape (Block Kit view_submission):
        {
          "type": "view_submission",
          "view": {
            "state": {
              "values": {
                "decision_block": {"decision_action": {"selected_option": {"value": "approve"}}},
                "diagnosis_block": {"diagnosis_action": {"selected_option": {"value": "correct"}}},
                ...
              }
            }
          },
          "user": {"id": "U12345ABC"}
        }
        """
        from airs_v2.memory.types import HumanFeedback

        values = (
            payload.get("view", {})
            .get("state", {})
            .get("values", {})
        )
        reviewer_id = payload.get("user", {}).get("id", "slack_user")

        def _extract(block_key: str, action_key: str, default: str = "") -> str:
            block = values.get(block_key, {})
            action = block.get(action_key, {})
            # Handle both selected_option and plain_text_input
            if "selected_option" in action:
                return action["selected_option"].get("value", default)
            if "value" in action:
                return action.get("value", default)
            return default

        def _extract_list(block_key: str, action_key: str) -> list[str]:
            raw = _extract(block_key, action_key, "")
            if not raw:
                return []
            return [s.strip() for s in raw.split("\n") if s.strip()]

        decision = _extract("decision_block", "decision_action", "approve")

        feedback = HumanFeedback(
            decision=decision,  # type: ignore[arg-type]
            reviewer_id=reviewer_id,
            diagnosis_accuracy=_extract(
                "diagnosis_block", "diagnosis_action", "correct"
            ),  # type: ignore[arg-type]
            correct_root_cause=_extract(
                "root_cause_block", "root_cause_action", ""
            ),
            causal_path_assessment=_extract(
                "causal_path_block", "causal_path_action", "correct"
            ),  # type: ignore[arg-type]
            action_appropriateness=_extract(
                "action_block", "action_action", "correct"
            ),  # type: ignore[arg-type]
            parameter_accuracy=_extract(
                "params_block", "params_action", "correct"
            ),  # type: ignore[arg-type]
            alternative_action=_extract("alt_action_block", "alt_action_action", ""),
            domain_expertise_notes=_extract("notes_block", "notes_action", ""),
            new_troubleshooting_steps=_extract_list(
                "steps_block", "steps_action"
            ),
            new_system_relationships=_extract_list(
                "relationships_block", "relationships_action"
            ),
            runbook_references=_extract_list(
                "runbooks_block", "runbooks_action"
            ),
            time_to_feedback_s=self._compute_latency(),
        )

        logger.info(
            "[FeedbackCollector] Parsed Slack payload: reviewer=%s decision=%s",
            reviewer_id,
            decision,
        )
        return feedback

    # ── CLI / JSON input parser ────────────────────────────────────────────────

    def parse_cli_input(self, json_str: str) -> object:
        """
        Parse a structured JSON string into HumanFeedback.

        Expected JSON keys mirror HumanFeedback fields directly.
        Unknown keys are silently ignored (extra='ignore' via Pydantic).

        Example input:
        {
          "decision": "approve",
          "diagnosis_accuracy": "correct",
          "domain_expertise_notes": "The Redis eviction was the actual trigger."
        }
        """
        from airs_v2.memory.types import HumanFeedback

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error("[FeedbackCollector] Invalid JSON input: %s", exc)
            raise

        data["time_to_feedback_s"] = self._compute_latency()
        feedback = HumanFeedback(**{
            k: v for k, v in data.items()
            if k in HumanFeedback.model_fields
        })

        logger.info(
            "[FeedbackCollector] Parsed CLI input: decision=%s", feedback.decision
        )
        return feedback

    # ── Mock feedback generator ────────────────────────────────────────────────

    def create_auto_feedback(
        self,
        auto_approve: bool = True,
        reviewer_id: str = "mock_reviewer",
        diagnosis_accuracy: str = "correct",
        action_appropriateness: str = "correct",
    ) -> object:
        """
        Create auto-generated HumanFeedback for testing and mock gateways.

        Parameters
        ----------
        auto_approve          : If True, decision="approve"; else "reject".
        reviewer_id           : Identifier for the mock reviewer.
        diagnosis_accuracy    : Feedback dimension for diagnosis quality.
        action_appropriateness: Feedback dimension for action quality.
        """
        from airs_v2.memory.types import HumanFeedback

        decision = "approve" if auto_approve else "reject"
        return HumanFeedback(
            decision=decision,  # type: ignore[arg-type]
            reviewer_id=reviewer_id,
            diagnosis_accuracy=diagnosis_accuracy,  # type: ignore[arg-type]
            action_appropriateness=action_appropriateness,  # type: ignore[arg-type]
            confidence_calibration="well_calibrated",
            time_to_feedback_s=0.0,
        )

    # ── Latency helper ─────────────────────────────────────────────────────────

    def _compute_latency(self) -> float:
        """Compute seconds elapsed since the prompt was sent."""
        if not self._prompt_sent_at:
            return 0.0
        try:
            sent = datetime.fromisoformat(self._prompt_sent_at)
            now = datetime.now(timezone.utc)
            return round((now - sent).total_seconds(), 2)
        except Exception:
            return 0.0
