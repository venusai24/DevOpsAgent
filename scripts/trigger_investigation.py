#!/usr/bin/env python3
"""
trigger_investigation.py — Manually trigger an investigation workflow.

Usage:
    python scripts/trigger_investigation.py \\
        --alert-id "alert-12345" \\
        --alert-name "HighErrorRate" \\
        --description "Error rate for payment-service exceeded 5% for 5 minutes" \\
        --service "payment-service" \\
        --namespace "production" \\
        --severity "critical"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog  # noqa: E402
from airs.config import settings  # noqa: E402
from airs.workflows.investigation_workflow import InvestigationWorkflow  # noqa: E402
from temporalio.client import Client  # noqa: E402

log = structlog.get_logger()


async def trigger(
    alert_id: str,
    alert_name: str,
    description: str,
    service: str,
    namespace: str,
    severity: str,
    wait: bool = False,
) -> None:
    """Connect to Temporal and start an InvestigationWorkflow."""
    client = await Client.connect(settings.temporal_address)

    log.info(
        "Triggering investigation workflow",
        alert_id=alert_id,
        alert_name=alert_name,
        service=service,
        severity=severity,
    )

    handle = await client.start_workflow(
        InvestigationWorkflow.run,
        args=[
            alert_id,
            alert_name,
            description,
            service,
            namespace,
            severity,
            {
                "alert_id": alert_id,
                "alert_name": alert_name,
                "description": description,
                "service": service,
                "namespace": namespace,
                "severity": severity,
            },
        ],
        id=f"investigation-{alert_id}",
        task_queue=settings.temporal_task_queue,
    )

    log.info(
        "Workflow started",
        workflow_id=handle.id,
        run_id=handle.result_run_id,
    )
    print("\n✅ Investigation started!")
    print(f"   Workflow ID: {handle.id}")
    print(f"   Monitor at:  http://localhost:8233/namespaces/default/workflows/{handle.id}")

    if wait:
        log.info("Waiting for workflow result...")
        result = await handle.result()
        print("\n📋 Investigation Result:")
        print(json.dumps(result.model_dump(mode="json"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger an AIRS investigation workflow")
    parser.add_argument("--alert-id", required=True, help="Unique alert identifier")
    parser.add_argument("--alert-name", required=True, help="Alert name (e.g., HighErrorRate)")
    parser.add_argument("--description", default="", help="Alert description")
    parser.add_argument("--service", required=True, help="Affected service name")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument("--severity", default="high", choices=["critical", "high", "medium", "low"])
    parser.add_argument("--wait", action="store_true", help="Wait for workflow completion")
    args = parser.parse_args()

    asyncio.run(trigger(
        alert_id=args.alert_id,
        alert_name=args.alert_name,
        description=args.description,
        service=args.service,
        namespace=args.namespace,
        severity=args.severity,
        wait=args.wait,
    ))


if __name__ == "__main__":
    main()
