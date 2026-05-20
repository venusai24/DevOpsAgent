from fastapi import APIRouter, Request, HTTPException, Form
import json
import logging
import asyncio
from langgraph.types import Command
from agent.orchestrator import build_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks/slack", tags=["Slack Webhooks"])

@router.post("/interactions")
async def handle_slack_interaction(payload: str = Form(...)):
    """
    Receives interactive callbacks from Slack when a user clicks 'Approve' or 'Reject'
    on the remediation plan Block Kit message.
    """
    try:
        data = json.loads(payload)
    except Exception as exc:
        logger.error(f"Failed to parse Slack interactive payload: {exc}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if data.get("type") == "block_actions":
        for action in data.get("actions", []):
            action_id = action.get("action_id")
            value = action.get("value")
            
            if action_id in ("approve_remediation", "reject_remediation"):
                # value is structured as "{decision}_{thread_id}"
                decision, thread_id = value.split("_", 1)
                approved = (decision == "approve")
                
                logger.info(f"Received Slack interactive decision: approved={approved} for thread_id={thread_id}")
                
                # In production, we'd trigger a background task or run this directly to resume the graph
                asyncio.create_task(_resume_graph(thread_id, approved))
                
                return {"status": "accepted"}
                
    return {"status": "ignored"}

async def _resume_graph(thread_id: str, approved: bool):
    """
    Resumes the LangGraph execution using the Postgres checkpointer.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        resume_cmd = Command(resume={"approved": approved})
        
        async with build_graph() as graph:
            await graph.ainvoke(resume_cmd, config)
            logger.info(f"Successfully resumed graph for thread_id={thread_id}")
    except Exception as e:
        logger.error(f"Failed to resume graph for thread_id={thread_id}: {e}")
