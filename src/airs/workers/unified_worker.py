"""
Unified Worker — Module 1.12.

Starts a Temporal worker that handles the InvestigationWorkflow and all
8 registered activities. Runs as a single process — one worker handles
all task queues (simplified Phase 1 deployment).

Phase 2: Separate worker pools for CPU-heavy (BERTScore) and I/O-heavy
(MCP tool) activities.

Usage:
    python -m airs.workers.unified_worker
    # or via: make worker
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

import structlog
from temporalio.client import Client
from temporalio.worker import Worker

from airs.activities import ALL_ACTIVITIES
from airs.config import settings
from airs.graph.state_graph import get_reasoning_graph
from airs.workflows.investigation_workflow import InvestigationWorkflow

log = structlog.get_logger()

_shutdown_event = asyncio.Event()


def _setup_signal_handlers() -> None:
    """Register SIGINT/SIGTERM handlers for graceful shutdown."""

    def _handle_shutdown(sig, frame):
        log.info("Received shutdown signal", signal=sig)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)


async def run_worker() -> None:
    """
    Main worker coroutine.
    Connects to Temporal server and starts the unified worker.
    """
    log.info(
        "Starting AIRS unified worker",
        temporal_host=settings.temporal_host,
        task_queue=settings.temporal_task_queue,
    )

    # Pre-compile LangGraph (expensive — do once at startup)
    log.info("Pre-compiling LangGraph reasoning graph...")
    get_reasoning_graph()
    log.info("LangGraph reasoning graph compiled")

    # Connect to Temporal
    try:
        client = await Client.connect(settings.temporal_host)
        log.info("Connected to Temporal server", host=settings.temporal_host)
    except Exception as e:
        log.error("Failed to connect to Temporal server", error=str(e))
        log.error("Ensure Temporal is running: make infra")
        sys.exit(1)

    # Create worker
    worker = Worker(
        client=client,
        task_queue=settings.temporal_task_queue,
        workflows=[InvestigationWorkflow],
        activities=ALL_ACTIVITIES,
        # Concurrency limits — conservative for Phase 1
        max_concurrent_activities=10,
        max_concurrent_workflow_tasks=5,
    )

    log.info(
        "Worker registered",
        workflows=["InvestigationWorkflow"],
        activities=[a.__name__ if hasattr(a, "__name__") else str(a) for a in ALL_ACTIVITIES],
    )

    # Run until shutdown signal
    run_task = asyncio.create_task(worker.run())
    shutdown_task = asyncio.create_task(_shutdown_event.wait())

    done, pending = await asyncio.wait(
        [run_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    log.info("Worker shutting down gracefully...")
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    log.info("Worker stopped")


def main() -> None:
    """Entry point for unified worker."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    _setup_signal_handlers()
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
