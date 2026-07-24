"""
run-incident.py — DevOps RCA Agent entry point with HITL pause/play support.

When the graph pauses for human input, a clear prompt is printed to the terminal
and the user picks an action + optional hint. The graph then resumes exactly where
it left off, with the hint injected into the agent's message context.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from devops_agent.core.container import AppContainer
from devops_agent.core.observability import init_observability
from devops_agent.orchestrator.state import InvestigationState

logging.basicConfig(level=logging.INFO)

investigation_id = str(uuid.uuid4())
trace_run_id = str(uuid.uuid4())
init_observability(run_id=investigation_id)

# 1. Initialize the Dependency Injection Container
app = AppContainer.get_instance()
app.initialize()  # type: ignore[no-untyped-call]

# 2. Define your exact Incident Time Window (Update the Date to match your CSVs)
start_time = datetime(2021, 3, 4, 10, 0, 0, tzinfo=UTC)
end_time = datetime(2021, 3, 4, 10, 30, 0, tzinfo=UTC)

# 3. Create the Initial Investigation State
initial_state = InvestigationState(
    investigation_id=investigation_id,
    current_trace_run_id=trace_run_id,
    investigation_state="active",
    time_range=(start_time, end_time),
    investigation_cluster=[],  # Agent will discover this via tools
    T0=start_time,             # Agent will refine this
    app_stats_path="/home/VenuSai/DevOpsAgent/incident_data/cluster_app_metrics.csv",
    metrics_path="/home/VenuSai/DevOpsAgent/incident_data/container_metrics.csv",
    logs_path="/home/VenuSai/DevOpsAgent/incident_data/cluster_incident_logs.csv",
    traces_path="/home/VenuSai/DevOpsAgent/incident_data/incident_traces.csv",
    declared_topology_graph={
        "IG01": [], "IG02": [], "MG01": [], "MG02": [], "Mysql01": [], "Mysql02": [],
        "Redis01": [], "Redis02": [], "ServiceTest1": [], "ServiceTest10": [],
        "ServiceTest11": [], "ServiceTest2": [], "ServiceTest3": [], "ServiceTest4": [],
        "ServiceTest5": [], "ServiceTest6": [], "ServiceTest7": [], "ServiceTest8": [],
        "ServiceTest9": [], "Tomcat01": [], "Tomcat02": [], "Tomcat03": [],
        "Tomcat04": [], "apache01": [], "apache02": [], "dockerA1": [],
        "dockerA2": [], "dockerB1": [], "dockerB2": []
    },
    explicit_symptoms={"dependencies_unknown": True}
)

config = {
    "configurable": {"thread_id": investigation_id},
    "run_name": "DevOps-Investigation",
    "run_id": trace_run_id
}

# ──────────────────────────────────────────────────────────────────────────────
# HITL helper — prompts the user in the terminal
# ──────────────────────────────────────────────────────────────────────────────

_ACTION_MAP = {
    "1": "RESTART_RCA",
    "2": "RESTART_TRIAGE",
    "3": "RESTART_CONTEXT",
    "4": "FORCE_CLOSE",
}

def _prompt_human(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Pretty-prints the HITL payload and asks the user for a response.
    Returns {"action": str, "hint": str}.
    """
    trigger    = payload.get("trigger", "UNKNOWN")
    description = payload.get("description", "")
    gaps       = payload.get("investigation_gaps", [])
    hypotheses = payload.get("surviving_hypotheses") or payload.get("top_hypotheses", [])
    scores     = payload.get("updated_scores", {})

    print("\n" + "=" * 70)
    print("🛑  AGENT PAUSED — HUMAN INPUT REQUIRED")
    print("=" * 70)
    print(f"  Trigger   : {trigger}")
    print(f"  Reason    : {description}")

    if hypotheses:
        print("\n  Active Hypotheses / Candidates:")
        for h in hypotheses[:5]:  # show at most 5
            name = h if isinstance(h, str) else h.get("name", str(h))
            score = scores.get(name, "?")
            print(f"    • {name}  (score: {score})")

    if gaps:
        print("\n  Known Gaps:")
        for g in gaps[:3]:
            reason = g.get("reason", str(g)) if isinstance(g, dict) else str(g)
            print(f"    • {reason}")

    print("\n  What should the agent do next?")
    print("    1. RESTART_RCA    — Retry evidence collection with your hint")
    print("    2. RESTART_TRIAGE — Re-examine blast radius")
    print("    3. RESTART_CONTEXT — Re-assemble context from scratch")
    print("    4. FORCE_CLOSE    — Generate report with current (partial) findings")
    print()

    # Action selection
    while True:
        choice = input("  Choice [1-4]: ").strip()
        if choice in _ACTION_MAP:
            action = _ACTION_MAP[choice]
            break
        print("  ⚠️  Please enter 1, 2, 3, or 4.")

    # Optional hint
    print()
    print("  Hint (optional — press Enter to skip):")
    print("  Examples: 'Check Redis evictions', 'Deployment happened at 10:05',")
    print("            'Ignore Mysql01 — it was being patched'")
    hint = input("  > ").strip()

    print()
    print(f"  ✅  Resuming with action={action}" + (f", hint='{hint}'" if hint else ""))
    print("=" * 70 + "\n")

    return {"action": action, "hint": hint}


# ──────────────────────────────────────────────────────────────────────────────
# Main investigation loop
# ──────────────────────────────────────────────────────────────────────────────

async def run_investigation() -> None:
    print(f"Starting Investigation: {investigation_id}\n")
    app.clock.start()

    current_input: Any = initial_state  # first invocation uses full initial state
    current_config = config

    try:
        while True:
            interrupted = False
            interrupt_payload = None

            async for event in app.graph.astream(current_input, config=current_config, stream_mode="updates"):
                for node_name, state_update in event.items():

                    # ── LangGraph interrupt signal ──────────────────────────
                    if node_name == "__interrupt__":
                        # state_update is a tuple/list of Interrupt objects.
                        # Guard: LangGraph may occasionally emit an empty sequence;
                        # treat that as a no-op rather than crashing on [0].
                        if isinstance(state_update, (list, tuple)):
                            if not state_update:
                                continue  # empty interrupt sequence — nothing to handle
                            interrupt_obj = state_update[0]
                        else:
                            interrupt_obj = state_update
                        interrupt_payload = getattr(interrupt_obj, "value", interrupt_obj)
                        interrupted = True
                        break  # stop consuming the stream; we must prompt the user

                    # ── Normal node completion ──────────────────────────────
                    print(f"✅ Completed Agent Node: {node_name}")

                    if isinstance(state_update, dict) and state_update.get("final_report"):
                        print("\n🔥 ROOT CAUSE REPORT 🔥")
                        print(state_update["final_report"])

                if interrupted:
                    break  # exit the async for loop to handle HITL

            # ── Handle the interrupt (pause → human prompt → play) ──────────
            if interrupted and interrupt_payload is not None:
                human_response = _prompt_human(interrupt_payload)

                # Resume by sending the human's structured response back.
                # The graph resumes from where it paused (before the HITL node).
                current_input = Command(resume=human_response)
                # config remains the same (same thread_id keeps the checkpoint)
                continue  # re-enter the while loop to keep streaming

            # ── No interrupt — graph finished naturally ─────────────────────
            break

    except Exception as e:
        print(f"\n❌ Investigation Failed: {e}")
        raise
    finally:
        app.clock.pause()
        print(f"\nTime Elapsed: {app.clock.elapsed_seconds():.1f}s")


asyncio.run(run_investigation())
