import os
import asyncio
from celery import Celery
from agent.orchestrator import build_graph
from agent.state import make_initial_state
from config import settings

celery_app = Celery(
    "airs_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

@celery_app.task(name="airs.tasks.run_agent_pipeline")
def run_agent_pipeline(alert_payload: dict, thread_id: str):
    """
    Background worker task to trigger the LangGraph execution safely outside the web request cycle.
    Because LangGraph is fundamentally async, we wrap the execution in asyncio.run.
    """
    asyncio.run(_run_graph(alert_payload, thread_id))

async def _run_graph(alert_payload: dict, thread_id: str):
    """
    Executes the AIRS graph workflow for a new incident.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = make_initial_state(alert_payload)
    
    async with build_graph() as graph:
        # In a real environment, we'd log the events to an external system, or let LangSmith handle it.
        async for event in graph.astream_events(initial_state, config, version="v2"):
            # Minimal logging for background worker context
            event_type = event.get("event", "")
            event_name = event.get("name", "")
            if event_type == "on_chain_start" and event_name in ("triage", "escalate", "investigate", "plan", "approval", "execute", "reject"):
                print(f"[Worker {thread_id}] Entering node: {event_name}")
