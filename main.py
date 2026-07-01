"""Main Entrypoint for DevOps Agent Orchestrator."""

import logging
import uuid

from devops_agent.core.container import AppContainer
from devops_agent.orchestrator.state import InvestigationState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_investigation(app_stats_tc: str, time_range: tuple, checkpointer_id: str | None = None):
    """
    Kicks off an autonomous RCA investigation.
    """
    app = AppContainer.get_instance()
    app.initialize()
    
    # 1. Initialize State
    inv_id = str(uuid.uuid4())
    initial_state = InvestigationState(
        investigation_id=inv_id,
        investigation_state="active",
        time_range=time_range,
        # Mock inputs mimicking an incoming alert
        investigation_cluster=[], 
        T0=time_range[0] # roughly
    )
    
    logger.info(f"Starting Investigation {inv_id} for TC: {app_stats_tc}")
    
    config = {"configurable": {"thread_id": checkpointer_id or inv_id}}
    
    # 2. Run Graph
    try:
        app.clock.start()
        
        for event in app.graph.stream(initial_state, config=config):
            for node_name, state_update in event.items():
                logger.info(f"Completed Node: {node_name}")
                # Real implementation would persist state_update via app.repository here
                
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        raise e
    finally:
        app.clock.pause()
        logger.info(f"Investigation complete. Wall-clock elapsed: {app.clock.elapsed_seconds()}s")

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=2)
    
    run_investigation(
        app_stats_tc="/api/v1/checkout",
        time_range=(start_time, end_time)
    )
