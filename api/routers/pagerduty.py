from fastapi import APIRouter, Request, HTTPException
import uuid
import logging
from worker.tasks import run_agent_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks/pagerduty", tags=["PagerDuty Webhooks"])

@router.post("")
async def handle_pagerduty_webhook(request: Request):
    """
    Receives incoming incident webhooks from PagerDuty, parses the incident data,
    and dispatches it to the background Celery worker queue to trigger the AIRS graph.
    """
    try:
        payload = await request.json()
    except Exception as exc:
        logger.error(f"Failed to parse PagerDuty webhook: {exc}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # In production, PagerDuty webhooks have a specific structure containing 'messages'
    # We will simulate the extraction here or assume it's passed in our specific format.
    incident_data = _parse_pagerduty_payload(payload)
    
    if not incident_data:
        return {"status": "ignored", "reason": "No actionable incident found in payload"}

    thread_id = str(uuid.uuid4())
    logger.info(f"Dispatching incident {incident_data.get('id')} to worker with thread_id={thread_id}")

    # Enqueue task in Celery
    run_agent_pipeline.delay(incident_data, thread_id)

    return {"status": "accepted", "incident_id": incident_data.get("id"), "thread_id": thread_id}

def _parse_pagerduty_payload(payload: dict) -> dict:
    """
    Extracts the relevant alert fields from a raw PagerDuty webhook event.
    For this demo, if the payload matches our internal schema, we return it directly,
    otherwise we map PagerDuty v3 Webhook fields to our internal incident structure.
    """
    # If it's already in our demo format (e.g., from a test script)
    if "id" in payload and "severity" in payload:
        return payload
        
    # PagerDuty V3 webhook parsing simulation
    if "event" in payload and payload["event"]["event_type"] == "incident.triggered":
        incident = payload["event"]["data"]
        return {
            "id": incident.get("id"),
            "title": incident.get("title", "Unknown PagerDuty Alert"),
            "service": incident.get("service", {}).get("summary", "unknown"),
            "severity": incident.get("urgency", "high").replace("high", "P0").replace("low", "P2"),
            "triggered_at": incident.get("created_at"),
            "source": "PagerDuty",
            "description": incident.get("description", "No description provided.")
        }
    
    return {}
