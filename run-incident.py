import asyncio
import logging
import uuid
from datetime import UTC, datetime

from devops_agent.core.container import AppContainer
from devops_agent.core.observability import init_observability
from devops_agent.orchestrator.state import InvestigationState

logging.basicConfig(level=logging.INFO)

investigation_id = str(uuid.uuid4())
init_observability(run_id=investigation_id)

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

config = {"configurable": {"thread_id": investigation_id}}

# 4. Execute the LangGraph Multi-Agent Orchestrator
print(f"Starting Investigation: {investigation_id}")
app.clock.start()

async def run_investigation():
    try:
        async for event in app.graph.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                print(f"✅ Completed Agent Node: {node_name}")
                
                # Print the final report if it was generated
                if "final_report" in state_update and state_update["final_report"]:
                    print("\n🔥 ROOT CAUSE REPORT 🔥")
                    print(state_update["final_report"])
                    
    except Exception as e:
        print(f"Investigation Failed: {e}")
    finally:
        app.clock.pause()
        print(f"Time Elapsed: {app.clock.elapsed_seconds()}s")

asyncio.run(run_investigation())
