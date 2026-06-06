"""
airs_v2/action/slack_gateway.py
================================

MockSlackGateway — Mock Human-Approval Webhook Client
-------------------------------------------------------

This module provides a **mock-only** Slack notification gateway.  It makes
no real HTTP calls and stores all sent messages in memory, enabling tests to
make deterministic assertions about what was routed for human approval.

In a production deployment this class would be replaced by a real webhook
client (e.g. using ``httpx`` to call the Slack Incoming Webhooks API or the
Slack Events API).  The interface is intentionally minimal so the swap is
a drop-in replacement.

Design invariants
-----------------
* ``send_approval_request`` is always async, mirroring the production shape.
* The returned ``message_ts`` is deterministic: ``"mock-ts-{action_id}"`` —
  tests can assert on it without any randomness.
* ``self._sent`` is a public-facing list; tests read it directly.
* The gateway logs at INFO level so approval requests appear in CI logs.

Usage
-----
::

    gateway = MockSlackGateway()
    ts = await gateway.send_approval_request(action, violations)
    assert len(gateway.sent_messages) == 1
    assert gateway.sent_messages[0]["action_id"] == action.action_id
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from airs_v2.action.types import PolicyViolation, RemediationAction

logger = logging.getLogger(__name__)


class MockSlackGateway:
    """
    Mock Slack webhook gateway for human-approval routing.

    All messages are stored in ``self.sent_messages`` for test inspection.
    No real network calls are made.

    Attributes
    ----------
    sent_messages:
        List of dicts representing each approval request sent.  Each dict
        contains: ``action_id``, ``action_kind``, ``target_namespace``,
        ``target_resource``, ``risk_level``, ``violations``, ``sent_at``,
        ``message_ts``.
    """

    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_approval_request(
        self,
        action: RemediationAction,
        violations: list[PolicyViolation],
    ) -> str:
        """
        Send a human-approval request for a high-risk action.

        Parameters
        ----------
        action:
            The action requiring human sign-off.
        violations:
            Any soft-gate violations that triggered the routing (may be empty
            if the action was routed purely by risk level via PE-R4).

        Returns
        -------
        str
            A deterministic mock message timestamp: ``"mock-ts-{action_id}"``.
        """
        message_ts = f"mock-ts-{action.action_id}"
        sent_at = datetime.now(timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "action_id": action.action_id,
            "action_kind": action.kind.value,
            "target_namespace": action.target_namespace,
            "target_resource": action.target_resource,
            "risk_level": action.risk_level.value,
            "estimated_blast_radius": action.estimated_blast_radius,
            "runbook_ref": action.runbook_ref,
            "violations": [v.model_dump() for v in violations],
            "sent_at": sent_at,
            "message_ts": message_ts,
        }

        self.sent_messages.append(payload)

        logger.info(
            "[SlackGateway] APPROVAL REQUEST action_id=%s kind=%s ns=%s "
            "risk=%s blast_radius=%d message_ts=%s",
            action.action_id,
            action.kind.value,
            action.target_namespace,
            action.risk_level.value,
            action.estimated_blast_radius,
            message_ts,
        )

        return message_ts

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored messages (useful between test runs)."""
        self.sent_messages.clear()

    @property
    def was_called(self) -> bool:
        """True if at least one approval request has been sent."""
        return len(self.sent_messages) > 0

    def messages_for_action(self, action_id: str) -> list[dict[str, Any]]:
        """Return all stored messages for the given action_id."""
        return [m for m in self.sent_messages if m["action_id"] == action_id]
