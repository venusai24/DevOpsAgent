"""
Command Line Interface (CLI) for DevOps RCA Agent.

This script can be run with arguments to provide paths and times directly,
or without arguments to trigger interactive prompts for the required inputs.
"""

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from devops_agent.core.container import AppContainer
from devops_agent.core.observability import init_observability
from devops_agent.orchestrator.state import InvestigationState

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default paths for convenience
DEFAULT_APP_METRICS = "/home/VenuSai/DevOpsAgent/incident_data/cluster_app_metrics.csv"
DEFAULT_CONTAINER_METRICS = "/home/VenuSai/DevOpsAgent/incident_data/container_metrics.csv"
DEFAULT_LOGS = "/home/VenuSai/DevOpsAgent/incident_data/cluster_incident_logs.csv"
DEFAULT_TRACES = "/home/VenuSai/DevOpsAgent/incident_data/incident_traces.csv"
DEFAULT_BASELINES = "/home/VenuSai/DevOpsAgent/incident_data/baselines.csv"

DEFAULT_START = "2021-03-04 10:00:00"
DEFAULT_END = "2021-03-04 10:30:00"

def get_input(prompt_text: str, default_val: str) -> str:
    """Prompt the user for input, falling back to a default value."""
    print(f"\n{prompt_text}")
    print(f"[{default_val}]")
    val = input("> ").strip()
    return val if val else default_val

def parse_datetime(dt_str: str) -> datetime:
    """Parse string to UTC datetime."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=UTC)
    except ValueError:
        print(f"Error: Could not parse '{dt_str}'. Please use format 'YYYY-MM-DD HH:MM:SS'.")
        sys.exit(1)

def interactive_prompt(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing arguments via interactive prompts."""
    print("=" * 60)
    print("🤖 DevOps Agentic System - Interactive CLI")
    print("=" * 60)
    
    if not args.start_time:
        args.start_time = get_input("Enter Incident Start Time (YYYY-MM-DD HH:MM:SS):", DEFAULT_START)
    if not args.end_time:
        args.end_time = get_input("Enter Incident End Time (YYYY-MM-DD HH:MM:SS):", DEFAULT_END)
    
    if not args.app_metrics:
        args.app_metrics = get_input("Enter App Metrics CSV Path:", DEFAULT_APP_METRICS)
    if not args.container_metrics:
        args.container_metrics = get_input("Enter Container Metrics CSV Path:", DEFAULT_CONTAINER_METRICS)
    if not args.logs:
        args.logs = get_input("Enter Logs CSV Path:", DEFAULT_LOGS)
    if not args.traces:
        args.traces = get_input("Enter Traces CSV Path:", DEFAULT_TRACES)
    if not args.baselines:
        args.baselines = get_input("Enter Baselines CSV Path:", DEFAULT_BASELINES)

    return args

_ACTION_MAP = {
    "1": "RESTART_RCA",
    "2": "RESTART_TRIAGE",
    "3": "RESTART_CONTEXT",
    "4": "FORCE_CLOSE",
}

def _prompt_human(payload: dict[str, Any]) -> dict[str, Any]:
    """Pretty-prints the HITL payload and asks the user for a response."""
    trigger = payload.get("trigger", "UNKNOWN")
    description = payload.get("description", "")
    gaps = payload.get("investigation_gaps", [])
    hypotheses = payload.get("surviving_hypotheses") or payload.get("top_hypotheses", [])
    scores = payload.get("updated_scores", {})

    print("\n" + "=" * 70)
    print("🛑  AGENT PAUSED — HUMAN INPUT REQUIRED")
    print("=" * 70)
    print(f"  Trigger   : {trigger}")
    print(f"  Reason    : {description}")

    if hypotheses:
        print("\n  Active Hypotheses / Candidates:")
        for h in hypotheses[:5]:
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
    print("    4. FORCE_CLOSE    — Generate report with current findings")
    print()

    while True:
        choice = input("  Choice [1-4]: ").strip()
        if choice in _ACTION_MAP:
            action = _ACTION_MAP[choice]
            break
        print("  ⚠️  Please enter 1, 2, 3, or 4.")

    print()
    print("  Hint (optional — press Enter to skip):")
    hint = input("  > ").strip()

    print()
    print(f"  ✅  Resuming with action={action}" + (f", hint='{hint}'" if hint else ""))
    print("=" * 70 + "\n")

    return {"action": action, "hint": hint}

async def run_investigation(args: argparse.Namespace) -> None:
    """Main execution loop for the agent."""
    start_dt = parse_datetime(args.start_time)
    end_dt = parse_datetime(args.end_time)

    investigation_id = str(uuid.uuid4())
    trace_run_id = str(uuid.uuid4())
    init_observability(run_id=investigation_id)

    app = AppContainer.get_instance()
    app.initialize()  # type: ignore[no-untyped-call]

    # Create Initial Investigation State
    initial_state = InvestigationState(
        investigation_id=investigation_id,
        current_trace_run_id=trace_run_id,
        investigation_state="active",
        time_range=(start_dt, end_dt),
        investigation_cluster=[],
        T0=start_dt,
        app_stats_path=args.app_metrics,
        metrics_path=args.container_metrics,
        logs_path=args.logs,
        traces_path=args.traces,
        baseline_registry_ref=args.baselines,
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

    print(f"\n🚀 Starting Investigation: {investigation_id}\n")
    app.clock.start()

    current_input: Any = initial_state
    current_config = config

    try:
        while True:
            interrupted = False
            interrupt_payload = None

            async for event in app.graph.astream(current_input, config=current_config, stream_mode="updates"):
                for node_name, state_update in event.items():
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
                        break

                    print(f"✅ Completed Agent Node: {node_name}")

                    if isinstance(state_update, dict) and state_update.get("final_report"):
                        print("\n🔥 ROOT CAUSE REPORT 🔥")
                        print(state_update["final_report"])

                if interrupted:
                    break

            if interrupted and interrupt_payload is not None:
                human_response = _prompt_human(interrupt_payload)
                current_input = Command(resume=human_response)
                continue

            break

    except Exception as e:
        print(f"\n❌ Investigation Failed: {e}")
        raise
    finally:
        app.clock.pause()
        print(f"\n⏱️  Time Elapsed: {app.clock.elapsed_seconds():.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="DevOps Agent Interactive CLI")
    parser.add_argument("--start-time", type=str, help="Incident Start Time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-time", type=str, help="Incident End Time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--app-metrics", type=str, help="Path to App Metrics CSV")
    parser.add_argument("--container-metrics", type=str, help="Path to Container Metrics CSV")
    parser.add_argument("--logs", type=str, help="Path to Logs CSV")
    parser.add_argument("--traces", type=str, help="Path to Traces CSV")
    parser.add_argument("--baselines", type=str, help="Path to Baselines CSV")

    args = parser.parse_args()

    # Fill any missing arguments via interactive prompts
    args = interactive_prompt(args)

    # Run the investigation loop
    asyncio.run(run_investigation(args))

if __name__ == "__main__":
    main()
