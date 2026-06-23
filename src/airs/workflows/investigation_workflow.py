"""
Investigation Workflow — Module 1.11.

Temporal workflow: The top-level orchestrator for a single incident investigation.
This is the durable execution backbone — every activity call is persisted in
the Temporal event history, enabling replay on worker restarts.

Workflow lifecycle:
  1. initialize_investigation  — build initial state from alert
  2. LOOP: run_reasoning_hop   — get next intent (EXECUTE_TOOL | QUERY_PLAYBOOK | DIAGNOSE | ESCALATE)
       ├─ EXECUTE_TOOL    → execute_mcp_tool → admit_evidence → verify_epistemic
       ├─ QUERY_PLAYBOOK  → retrieve_playbook_context → run_reasoning_hop (with context injected)
       ├─ DIAGNOSE        → produce_diagnosis → return InvestigationResult
       └─ ESCALATE        → produce_escalation → return InvestigationResult
  3. Return InvestigationResult

Design decisions:
  - Each iteration of the loop is ONE reasoning hop
  - Activities run with retries (Temporal handles failure recovery)
  - State is serialised to Temporal history after every activity
  - Max 50 hops (from config) before forced ESCALATE
  - Query playbook does NOT count as a reasoning hop (no hop counter increment)
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    # These imports need to be visible to the workflow sandbox but are
    # imported in a pass-through manner (Temporal sandboxing requirement)
    from airs.activities import (
        ALL_ACTIVITIES,
        admit_evidence_candidate,
        execute_mcp_tool,
        initialize_investigation,
        produce_diagnosis,
        produce_escalation,
        retrieve_playbook_context,
        run_reasoning_hop_activity,
        verify_epistemic_state,
    )
    from airs.models.intents import ExecutionIntent, IntentAction
    from airs.models.investigation import InvestigationState
    from airs.models.results import InvestigationResult

log = logging.getLogger(__name__)

# Temporal activity retry policy — conservative for production
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=3,
)

# Generous timeout for tool execution (some MCP servers are slow)
_TOOL_ACTIVITY_TIMEOUT = timedelta(minutes=5)
_LLM_ACTIVITY_TIMEOUT = timedelta(minutes=3)
_SHORT_ACTIVITY_TIMEOUT = timedelta(seconds=30)


@workflow.defn(name="InvestigationWorkflow")
class InvestigationWorkflow:
    """
    Durable investigation workflow orchestrating the full AIRS agent loop.

    Workflow inputs:
        alert_id, alert_name, description, service, namespace,
        severity, raw_alert_payload

    Workflow output:
        InvestigationResult (RESOLVED or ESCALATED)
    """

    def __init__(self) -> None:
        self._state: Optional[InvestigationState] = None
        self._playbook_context: str = ""

    @workflow.run
    async def run(
        self,
        alert_id: str,
        alert_name: str,
        description: str,
        service: str,
        namespace: str,
        severity: str,
        raw_alert_payload: dict,
    ) -> InvestigationResult:
        """Main workflow entry point."""
        from airs.config import settings

        workflow.logger.info(
            "Investigation workflow started: alert=%s service=%s",
            alert_name, service,
        )

        # ── Step 1: Initialize investigation ──────────────────────────────────
        self._state = await workflow.execute_activity(
            initialize_investigation,
            args=[alert_id, alert_name, description, service, namespace, severity, raw_alert_payload],
            start_to_close_timeout=_SHORT_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )

        # ── Step 2: Investigation loop ────────────────────────────────────────
        max_hops = settings.max_hops

        for hop in range(max_hops):
            workflow.logger.info(
                "Investigation hop %d/%d: investigation=%s",
                hop, max_hops, alert_id,
            )

            # ── Run reasoning hop ─────────────────────────────────────────────
            if self._playbook_context:
                # Inject playbook context into state before reasoning
                self._state = _inject_playbook_context(self._state, self._playbook_context)
                self._playbook_context = ""  # Consume after injection

            intent, updated_state = await workflow.execute_activity(
                run_reasoning_hop_activity,
                args=[self._state],
                start_to_close_timeout=_LLM_ACTIVITY_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
            self._state = updated_state

            # ── Route on intent ───────────────────────────────────────────────
            action = intent.action

            if action == IntentAction.DIAGNOSE:
                workflow.logger.info("Diagnosis action at hop %d", hop)
                result = await workflow.execute_activity(
                    produce_diagnosis,
                    args=[self._state],
                    start_to_close_timeout=_LLM_ACTIVITY_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
                return result

            elif action == IntentAction.ESCALATE:
                reason = intent.reasoning or f"Escalation at hop {hop}"
                workflow.logger.warning("Escalation at hop %d: %s", hop, reason)
                result = await workflow.execute_activity(
                    produce_escalation,
                    args=[self._state, reason],
                    start_to_close_timeout=_LLM_ACTIVITY_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
                return result

            elif action == IntentAction.QUERY_PLAYBOOK:
                if intent.playbook_query is not None:
                    # Retrieve playbook context — does NOT count as a hop
                    self._playbook_context = await workflow.execute_activity(
                        retrieve_playbook_context,
                        args=[intent.playbook_query],
                        start_to_close_timeout=_SHORT_ACTIVITY_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    workflow.logger.info(
                        "Playbook retrieved at hop %d: %d chars",
                        hop, len(self._playbook_context),
                    )
                # Don't increment hop counter — continue loop
                continue

            elif action == IntentAction.EXECUTE_TOOL:
                if not intent.tool_specs:
                    workflow.logger.error("EXECUTE_TOOL intent missing tool_specs at hop %d", hop)
                    continue

                # ── Execute MCP tools concurrently ────────────────────────────
                import asyncio
                futures = [
                    workflow.execute_activity(
                        execute_mcp_tool,
                        args=[
                            alert_id,
                            hop,
                            spec,
                            service,
                            namespace,
                        ],
                        start_to_close_timeout=_TOOL_ACTIVITY_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    for spec in intent.tool_specs
                ]
                candidates = await asyncio.gather(*futures)

                # ── Admit evidence to context & verify sequentially ───────────
                pasc_violation = False
                for i, candidate in enumerate(candidates):
                    spec = intent.tool_specs[i]

                    self._state, _metrics = await workflow.execute_activity(
                        admit_evidence_candidate,
                        args=[self._state, candidate],
                        start_to_close_timeout=_SHORT_ACTIVITY_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )

                    # ── Epistemic verification ────────────────────────────────
                    # Note: logprobs are not available from all LLMs; use [] as fallback
                    self._state, pasc_ok = await workflow.execute_activity(
                        verify_epistemic_state,
                        args=[
                            self._state,
                            [],  # logprobs — populated by LLM response in Phase 2
                            "",  # llm_interpretation — from previous reasoning output
                            str(candidate.filtered_content),
                            spec.tier.value,
                        ],
                        start_to_close_timeout=_SHORT_ACTIVITY_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )

                    if not pasc_ok:
                        pasc_violation = True
                        workflow.logger.warning(
                            "PASC coverage violation at hop %d for tool %s — aborting tool chain",
                            hop, spec.tool_name
                        )
                        break

                # Increment hop counter once per reasoning cycle
                self._state = self._state.increment_hop()

        # ── Max hops reached without terminal action ──────────────────────────
        workflow.logger.warning(
            "Max hops (%d) reached without DIAGNOSE/ESCALATE — forcing escalation",
            max_hops,
        )
        result = await workflow.execute_activity(
            produce_escalation,
            args=[self._state, f"Max hops ({max_hops}) reached without convergence"],
            start_to_close_timeout=_LLM_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )
        return result

    @workflow.signal
    async def inject_context(self, additional_context: str) -> None:
        """
        Temporal signal: inject additional context from external systems.
        Allows operators to provide context during an active investigation.
        """
        self._playbook_context = (self._playbook_context or "") + "\n" + additional_context
        workflow.logger.info("Context injected via signal (%d chars)", len(additional_context))

    @workflow.query
    def get_current_state(self) -> dict:
        """Temporal query: return current investigation state summary."""
        if self._state is None:
            return {"status": "not_started"}
        return {
            "investigation_id": self._state.investigation_id,
            "hop_count": self._state.total_hop_count,
            "status": self._state.status.value,
            "active_evidence_count": len(self._state.insight_tiers.active),
            "missing_mass": self._state.pursuit_state.current_missing_mass,
            "supermartingale": self._state.risk_state.supermartingale_value,
            "leading_hypothesis": (
                self._state.hypotheses[0].statement[:100]
                if self._state.hypotheses else None
            ),
        }


def _inject_playbook_context(
    state: InvestigationState, context: str
) -> InvestigationState:
    """
    Inject playbook context string into the state's context buffer.
    This is stored in a reserved field so the reasoning activity can
    incorporate it into the next LLM prompt.
    """
    # Store playbook context in the state's alert description extension
    # Phase 2: Dedicated playbook_context field in InvestigationState
    existing = state.alert.raw_payload.get("_playbook_context", "")
    updated_payload = {
        **state.alert.raw_payload,
        "_playbook_context": (existing + "\n" + context).strip(),
    }
    updated_alert = state.alert.model_copy(update={"raw_payload": updated_payload})
    return state.model_copy(update={"alert": updated_alert})
