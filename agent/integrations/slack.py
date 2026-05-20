import logging
import httpx
from config import settings
from agent.state import RemediationPlan

logger = logging.getLogger(__name__)

async def send_slack_approval_request(plan_md: str, risk_label: str, thread_id: str, is_high_risk: bool):
    """
    Sends a Block Kit message to a Slack channel to request approval for a remediation plan.
    """
    if not settings.SLACK_BOT_TOKEN or not settings.SLACK_CHANNEL_ID:
        logger.info("[Simulated Slack] Would have sent approval request to Slack.")
        return

    # In a real implementation, we would construct a beautiful Slack Block Kit payload.
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🚨 [AIRS] REMEDIATION PLAN AWAITING APPROVAL ({risk_label})",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": plan_md[:2000] # truncate if too long
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Approve Remediation",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": f"approve_{thread_id}",
                    "action_id": "approve_remediation"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Reject / Abort",
                        "emoji": True
                    },
                    "style": "danger",
                    "value": f"reject_{thread_id}",
                    "action_id": "reject_remediation"
                }
            ]
        }
    ]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {settings.SLACK_BOT_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "channel": settings.SLACK_CHANNEL_ID,
                    "blocks": blocks,
                    "text": "New Incident Remediation Plan Awaiting Approval"
                }
            )
            response.raise_for_status()
            logger.info("Sent Slack approval request.")
    except Exception as e:
        logger.error(f"Failed to send Slack message: {e}")
