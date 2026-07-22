import asyncio
import logging
import uuid
from datetime import UTC, datetime

from devops_agent.core.container import AppContainer
from devops_agent.core.observability import init_observability
from devops_agent.orchestrator.state import InvestigationState

logging.basicConfig(level=logging.INFO)

investigation_id = str(uuid.uuid4())
trace_run_id = str(uuid.uuid4())
init_observability(run_id=investigation_id)

# --- STRICT DEBUG INJECTION ---
import numpy as np
from langgraph.checkpoint.memory import MemorySaver

_original_put = MemorySaver.put
def _debug_put(self, config, checkpoint, metadata, new_versions):
    def check_for_numpy(obj, path="root"):
        if isinstance(obj, np.float64):
            print(f"\\n🚨 [DEBUG] FOUND numpy.float64 AT PATH: {path} (value: {obj})\\n")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                check_for_numpy(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_for_numpy(v, f"{path}[{i}]")
        elif isinstance(obj, tuple):
            for i, v in enumerate(obj):
                check_for_numpy(v, f"{path}[{i}]")
    
    check_for_numpy(checkpoint)
    try:
        return _original_put(self, config, checkpoint, metadata, new_versions)
    except TypeError as e:
        if "numpy.float64" in str(e):
            print(f"\\n🚨 [FATAL CHECKPOINT ERROR] {e}")
            print(f"🚨 [FATAL CHECKPOINT ERROR] Dumping problematic checkpoint keys:")
            for k, v in checkpoint["channel_values"].items():
                print(f"   - Key: {k}, Type: {type(v)}")
        raise e

MemorySaver.put = _debug_put
# ------------------------------

# 1. Initialize the Dependency Injection Container
app = AppContainer.get_instance()
app.initialize()

# 2. Define your exact Incident Time Window (Update the Date to match your CSVs)
# Note: Since your CSVs are pre-filtered, the exact date matters less, 
# but DuckDB relies on these timestamps for its WHERE clauses!
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
    explicit_symptoms={"dependencies_unknown": True},
    match_results=[],
    semantic_facts=[],
    playbook_verdicts=[]
)

config = {
    "configurable": {"thread_id": investigation_id},
    "run_name": "DevOps-Investigation",
    "run_id": trace_run_id
}

# 4. Execute the LangGraph Multi-Agent Orchestrator
print(f"Starting Investigation: {investigation_id}")
app.clock.start()

async def run_investigation():
    from langgraph.types import Command
    from typing import Any
    try:
        command_or_state: Any = initial_state
        while True:
            async for event in app.graph.astream(command_or_state, config=config):
                for node_name, state_update in event.items():
                    print(f"✅ Completed Agent Node: {node_name}")
                    if node_name == "triage":
                        print(f"TRIAGE STATE: {state_update}")
                    
                    # Print the final report if it was generated
                    if "final_report" in state_update and state_update["final_report"]:
                        print("\n🔥 ROOT CAUSE REPORT 🔥")
                        print(state_update["final_report"])
            
            # Check if execution paused due to an interrupt() or HITL breakpoint
            state = app.graph.get_state(config)
            if not state.next:
                # Execution finished completely
                print("\n=== FINAL INVESTIGATION RESULTS ===")
                final_val = state.values
                print(f"\n📝 Narrative Summary:\n{final_val.get('narrative_summary', 'None provided')}")
                print(f"\n🎯 Root Cause Candidate:\n{final_val.get('root_cause_candidate', 'None provided')}")
                print(f"\n🔥 Final Report:\n{final_val.get('final_report', 'None provided')}")
                break 
                
            print(f"\n⚠️ Graph Execution Paused at nodes: {state.next}")
            
            if "ambiguity_node" in state.next:
                print("\n[Ambiguity Detected]: The agent requires human guidance.")
                hint = input("Enter hint (or type 'manual' for override, 'resume' to ignore): ")
                if hint.lower() == 'manual':
                    cause = input("Enter manual root cause override: ")
                    command_or_state = Command(resume={"action": "manual_root_cause", "root_cause": cause})
                elif hint.lower() == 'resume':
                    command_or_state = Command(resume={"action": "ignore"})
                else:
                    command_or_state = Command(resume={"action": "hint", "hint": hint})
            else:
                user_input = input("Press Enter to resume execution... ")
                command_or_state = Command(resume=user_input if user_input else "resume")
                    
    except Exception as e:
        print(f"Investigation Failed: {e}")
    finally:
        app.clock.pause()
        print(f"Time Elapsed: {app.clock.elapsed_seconds()}s")

asyncio.run(run_investigation())
