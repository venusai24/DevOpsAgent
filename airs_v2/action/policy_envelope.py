"""
airs_v2/action/policy_envelope.py
==================================

PolicyEnvelope — Deterministic Five-Rule Remediation Action Gate
-----------------------------------------------------------------

The PolicyEnvelope is the mandatory validation layer that every
``RemediationPlan`` must pass through before any action is routed or approved.

It is a **pure rule engine**: no LLM calls, no network I/O, no randomness.
Given the same inputs it always produces the same outputs.

The Five Rules
--------------

PE-R1  NAMESPACE_ALLOWLIST
    ``action.target_namespace`` must be in the configured ``allowed_namespaces``
    set.  Default set: ``{"production", "staging", "default"}``.
    Deliberately excludes "monitoring", "kube-system", "kube-public", and
    "cert-manager" to prevent the agent from touching infrastructure namespaces.

PE-R2  BLAST_RADIUS_CAP
    ``action.estimated_blast_radius`` must be ≤ ``MAX_BLAST_RADIUS`` (default 50)
    for LOW/MEDIUM/HIGH risk actions.  For CRITICAL risk actions the cap is
    tighter: ≤ ``MAX_CRITICAL_BLAST_RADIUS`` (default 10).

PE-R3  BUSINESS_HOURS
    Actions of kind ``exec_script`` or ``drain_node`` are only allowed during
    business hours (09:00–17:00 on weekdays) in the configured timezone.
    The timezone is read from the ``POLICY_TIMEZONE`` environment variable,
    defaulting to UTC.  Pass ``action.parameters["emergency"] = True`` to
    override this rule for genuine out-of-hours incidents.

PE-R4  RISK_ROUTING
    Actions with ``effective_risk_level == CRITICAL`` are REJECTED outright.
    Actions with ``effective_risk_level == HIGH`` are routed to human approval
    (``PENDING_HUMAN``) instead of being auto-approved.
    LOW and MEDIUM actions (that pass R1–R3) receive ``AUTO_APPROVED``.

PE-R5  EXEC_SCRIPT_ALWAYS_HIGH
    Any action whose ``kind == exec_script`` is treated as ``HIGH`` risk
    regardless of the declared ``risk_level``.  This prevents policy bypass
    via mislabeling.  The ``effective_risk_level`` field in ``PolicyDecision``
    reflects the post-R5 level.

Evaluation order
----------------
Rules are evaluated in order R1 → R5.  R1–R3 are hard gates that produce
``REJECTED`` immediately (fail-fast; remaining rules are skipped).  R4 and R5
set the routing decision but do not short-circuit further rule evaluation.

Usage
-----
::

    from airs_v2.action.policy_envelope import PolicyEnvelope
    from airs_v2.action.types import RemediationPlan

    envelope = PolicyEnvelope()
    decisions = envelope.evaluate(plan, now=datetime.now(timezone.utc))
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, time as dt_time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from airs_v2.action.types import (
    ActionKind,
    DecisionOutcome,
    PolicyDecision,
    PolicyViolation,
    RemediationAction,
    RemediationPlan,
    RiskLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule identifiers (machine-readable)
# ---------------------------------------------------------------------------

PE_R1 = "PE-R1"
PE_R2 = "PE-R2"
PE_R3 = "PE-R3"
PE_R4 = "PE-R4"
PE_R5 = "PE-R5"

PE_R1_NAME = "NAMESPACE_ALLOWLIST"
PE_R2_NAME = "BLAST_RADIUS_CAP"
PE_R3_NAME = "BUSINESS_HOURS"
PE_R4_NAME = "RISK_ROUTING"
PE_R5_NAME = "EXEC_SCRIPT_ALWAYS_HIGH"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Namespaces where autonomous remediation is permitted.
#: Tight by design — "monitoring", "kube-system", etc. are intentionally absent.
DEFAULT_ALLOWED_NAMESPACES: frozenset[str] = frozenset(
    {"production", "staging", "default"}
)

#: Maximum pods that may be affected for LOW/MEDIUM/HIGH actions.
MAX_BLAST_RADIUS: int = 50

#: Maximum pods that may be affected for CRITICAL-labelled actions (tighter).
MAX_CRITICAL_BLAST_RADIUS: int = 10

#: Business hours window — half-open interval [BUSINESS_HOURS_START, BUSINESS_HOURS_END).
BUSINESS_HOURS_START: dt_time = dt_time(9, 0)   # 09:00 inclusive
BUSINESS_HOURS_END: dt_time = dt_time(17, 0)    # 17:00 exclusive

#: The set of ISO weekday numbers considered business days (Mon=1 … Sun=7).
BUSINESS_DAYS: frozenset[int] = frozenset({1, 2, 3, 4, 5})  # Mon–Fri

#: Actions that are subject to the business-hours restriction.
BUSINESS_HOURS_RESTRICTED_KINDS: frozenset[ActionKind] = frozenset(
    {ActionKind.EXEC_SCRIPT, ActionKind.DRAIN_NODE}
)


# ---------------------------------------------------------------------------
# Timezone helper
# ---------------------------------------------------------------------------


def _resolve_timezone() -> ZoneInfo:
    """
    Resolve the policy timezone from the ``POLICY_TIMEZONE`` environment variable.

    Falls back to UTC silently if the variable is unset or contains an invalid
    IANA timezone name.  This prevents a misconfigured environment from crashing
    the policy engine — a fatal failure in the safety layer is worse than
    silently defaulting to a conservative UTC baseline.
    """
    tz_name = os.environ.get("POLICY_TIMEZONE", "UTC")
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning(
            "[PolicyEnvelope] Unknown POLICY_TIMEZONE=%r, falling back to UTC",
            tz_name,
        )
        return ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# PolicyEnvelope
# ---------------------------------------------------------------------------


class PolicyEnvelope:
    """
    Deterministic five-rule policy gate for remediation plans.

    Parameters
    ----------
    allowed_namespaces:
        Override the default namespace allowlist.  If ``None``, uses
        ``DEFAULT_ALLOWED_NAMESPACES``.
    max_blast_radius:
        Override the pod-count cap for non-CRITICAL actions.
    max_critical_blast_radius:
        Override the pod-count cap for CRITICAL-labelled actions.
    """

    def __init__(
        self,
        allowed_namespaces: frozenset[str] | set[str] | None = None,
        max_blast_radius: int = MAX_BLAST_RADIUS,
        max_critical_blast_radius: int = MAX_CRITICAL_BLAST_RADIUS,
    ) -> None:
        self._allowed_ns: frozenset[str] = (
            frozenset(allowed_namespaces)
            if allowed_namespaces is not None
            else DEFAULT_ALLOWED_NAMESPACES
        )
        self._max_br = max_blast_radius
        self._max_critical_br = max_critical_blast_radius
        self._tz = _resolve_timezone()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        plan: RemediationPlan,
        *,
        now: datetime | None = None,
    ) -> list[PolicyDecision]:
        """
        Evaluate every action in *plan* through the five-rule pipeline.

        Parameters
        ----------
        plan:
            The remediation plan to validate.
        now:
            The reference timestamp for business-hours evaluation.  Defaults
            to ``datetime.now(timezone.utc)``.  Inject an explicit value in
            tests to make time-dependent rules deterministic.

        Returns
        -------
        list[PolicyDecision]
            One ``PolicyDecision`` per action, in plan order.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        decisions: list[PolicyDecision] = []
        for action in plan.actions:
            decision = self._evaluate_action(action, now=now)
            decisions.append(decision)
            logger.info(
                "[PolicyEnvelope] action_id=%s kind=%s ns=%s risk=%s "
                "effective_risk=%s decision=%s",
                action.action_id,
                action.kind.value,
                action.target_namespace,
                action.risk_level.value,
                decision.effective_risk_level.value,
                decision.decision.value,
            )
        return decisions

    # ------------------------------------------------------------------
    # Internal — per-action evaluation
    # ------------------------------------------------------------------

    def _evaluate_action(
        self, action: RemediationAction, *, now: datetime
    ) -> PolicyDecision:
        """Apply rules R1–R5 to a single action and return its PolicyDecision."""
        violations: list[PolicyViolation] = []

        # ── PE-R5: exec_script always HIGH (apply first, affects R4 routing) ──
        effective_risk = self._apply_r5(action)

        # ── PE-R1: Namespace allowlist ─────────────────────────────────────
        r1_violation = self._check_r1(action)
        if r1_violation:
            violations.append(r1_violation)
            return self._decision(
                action, effective_risk, DecisionOutcome.REJECTED, violations
            )

        # ── PE-R2: Blast radius cap ────────────────────────────────────────
        r2_violation = self._check_r2(action, effective_risk)
        if r2_violation:
            violations.append(r2_violation)
            return self._decision(
                action, effective_risk, DecisionOutcome.REJECTED, violations
            )

        # ── PE-R3: Business hours ──────────────────────────────────────────
        r3_violation = self._check_r3(action, now=now)
        if r3_violation:
            violations.append(r3_violation)
            return self._decision(
                action, effective_risk, DecisionOutcome.REJECTED, violations
            )

        # ── PE-R4: Risk routing ────────────────────────────────────────────
        r4_violation, outcome = self._apply_r4(action, effective_risk)
        if r4_violation:
            violations.append(r4_violation)
        return self._decision(action, effective_risk, outcome, violations)

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _apply_r5(self, action: RemediationAction) -> RiskLevel:
        """
        PE-R5: exec_script actions are always treated as HIGH risk.

        Returns the *effective* risk level (may be upgraded from the declared
        level).  Never downgrades.
        """
        if action.kind == ActionKind.EXEC_SCRIPT:
            declared = action.risk_level
            if declared in (RiskLevel.LOW, RiskLevel.MEDIUM):
                logger.debug(
                    "[PolicyEnvelope] PE-R5 upgrade: action_id=%s %s→HIGH",
                    action.action_id,
                    declared.value,
                )
                return RiskLevel.HIGH
        return action.risk_level

    def _check_r1(self, action: RemediationAction) -> PolicyViolation | None:
        """PE-R1: Namespace must be in the allowlist."""
        if action.target_namespace not in self._allowed_ns:
            return PolicyViolation(
                rule_id=PE_R1,
                rule_name=PE_R1_NAME,
                action_id=action.action_id,
                reason=(
                    f"Namespace '{action.target_namespace}' is not in the "
                    f"allowed namespace set {sorted(self._allowed_ns)}. "
                    f"Autonomous remediation is restricted to application "
                    f"namespaces only."
                ),
            )
        return None

    def _check_r2(
        self, action: RemediationAction, effective_risk: RiskLevel
    ) -> PolicyViolation | None:
        """PE-R2: Blast radius must not exceed the configured cap."""
        cap = (
            self._max_critical_br
            if effective_risk == RiskLevel.CRITICAL
            else self._max_br
        )
        if action.estimated_blast_radius > cap:
            return PolicyViolation(
                rule_id=PE_R2,
                rule_name=PE_R2_NAME,
                action_id=action.action_id,
                reason=(
                    f"Estimated blast radius of {action.estimated_blast_radius} pods "
                    f"exceeds the cap of {cap} pods for risk level "
                    f"'{effective_risk.value}'. Reduce scope or obtain manual approval."
                ),
            )
        return None

    def _check_r3(
        self, action: RemediationAction, *, now: datetime
    ) -> PolicyViolation | None:
        """PE-R3: Restricted action kinds are blocked outside business hours."""
        if action.kind not in BUSINESS_HOURS_RESTRICTED_KINDS:
            return None

        # Emergency override: bypass time restriction
        if action.parameters.get("emergency") is True:
            logger.debug(
                "[PolicyEnvelope] PE-R3 bypassed via emergency flag: action_id=%s",
                action.action_id,
            )
            return None

        # Convert `now` to the policy timezone
        local_now = now.astimezone(self._tz)
        weekday = local_now.isoweekday()  # Mon=1 … Sun=7
        local_time = local_now.time().replace(tzinfo=None)

        in_business_hours = (
            weekday in BUSINESS_DAYS
            and BUSINESS_HOURS_START <= local_time < BUSINESS_HOURS_END
        )

        if not in_business_hours:
            tz_name = str(self._tz)
            return PolicyViolation(
                rule_id=PE_R3,
                rule_name=PE_R3_NAME,
                action_id=action.action_id,
                reason=(
                    f"Action kind '{action.kind.value}' is restricted to business "
                    f"hours ({BUSINESS_HOURS_START.strftime('%H:%M')}–"
                    f"{BUSINESS_HOURS_END.strftime('%H:%M')} {tz_name}, Mon–Fri). "
                    f"Current time: {local_now.strftime('%A %H:%M %Z')}. "
                    f"Set parameters['emergency']=True to override."
                ),
            )
        return None

    def _apply_r4(
        self, action: RemediationAction, effective_risk: RiskLevel
    ) -> tuple[PolicyViolation | None, DecisionOutcome]:
        """
        PE-R4: Route based on effective risk level.

        CRITICAL → REJECTED (hard gate, no human-approval path).
        HIGH     → PENDING_HUMAN (route to Slack for human review).
        LOW/MED  → AUTO_APPROVED (no violation emitted).

        Returns a (violation | None, outcome) tuple so callers can attach the
        violation to the decision record for full auditability.
        """
        if effective_risk == RiskLevel.CRITICAL:
            violation = PolicyViolation(
                rule_id=PE_R4,
                rule_name=PE_R4_NAME,
                action_id=action.action_id,
                reason=(
                    f"Action '{action.action_id}' has risk level CRITICAL. "
                    f"CRITICAL-risk actions are unconditionally rejected — there is "
                    f"no human-approval path. Reduce scope, split into smaller actions, "
                    f"or obtain an explicit change-management approval outside this system."
                ),
            )
            return violation, DecisionOutcome.REJECTED

        if effective_risk == RiskLevel.HIGH:
            violation = PolicyViolation(
                rule_id=PE_R4,
                rule_name=PE_R4_NAME,
                action_id=action.action_id,
                reason=(
                    f"Action '{action.action_id}' has effective risk level HIGH "
                    f"(declared: {action.risk_level.value}). Routed to human approval "
                    f"via Slack — autonomous execution is not permitted."
                ),
            )
            return violation, DecisionOutcome.PENDING_HUMAN

        return None, DecisionOutcome.AUTO_APPROVED

    # ------------------------------------------------------------------
    # Helper — build PolicyDecision
    # ------------------------------------------------------------------

    @staticmethod
    def _decision(
        action: RemediationAction,
        effective_risk: RiskLevel,
        outcome: DecisionOutcome,
        violations: list[PolicyViolation],
    ) -> PolicyDecision:
        return PolicyDecision(
            action_id=action.action_id,
            action_kind=action.kind.value,
            effective_risk_level=effective_risk,
            approved=(outcome == DecisionOutcome.AUTO_APPROVED),
            decision=outcome,
            violations=violations,
        )
