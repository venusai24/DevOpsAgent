"""LangSmith utility functions for tracing extraction and feedback."""

import asyncio
import json
import logging

from langsmith import Client

logger = logging.getLogger(__name__)

async def fetch_run_tree_async(run_id: str) -> str:
    """
    Asynchronously fetches the full run trace from LangSmith for the given run_id.
    Formats the trace (LLM inputs/outputs, Tool calls, Errors) into a concise
    Markdown string suitable for LLM context window.
    """
    def _fetch():
        client = Client()
        # Fetch the root run and its descendants
        runs = list(client.list_runs(trace_id=run_id))
        if not runs:
            return f"No trace found for Run ID: {run_id}"

        # Sort runs by start_time to maintain causal order
        runs.sort(key=lambda r: (r.start_time or 0))
        
        trace_summary = [f"# Trace Log for Trace ID {run_id}"]
        for run in runs:
            run_type = run.run_type
            name = run.name
            error = run.error
            inputs = json.dumps(run.inputs, default=str) if run.inputs else "None"
            
            outputs = json.dumps(run.outputs, default=str) if run.outputs else "None"
                
            elapsed = 0.0
            if run.end_time and run.start_time:
                elapsed = (run.end_time - run.start_time).total_seconds() * 1000
                
            trace_summary.append(f"## {run_type.upper()}: {name} ({elapsed:.1f}ms)")
            if error:
                trace_summary.append(f"**ERROR**: {error}")
            trace_summary.append(f"**Inputs**: {inputs}")
            trace_summary.append(f"**Outputs**: {outputs}")
            trace_summary.append("---")
            
        return "\\n".join(trace_summary)

    try:
        # Offload the blocking HTTP request to a background thread
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error(f"Failed to fetch run tree for {run_id}: {e}")
        return f"Error retrieving trace: {e}"

async def submit_run_feedback_async(run_id: str, key: str, score: float, comment: str = "") -> None:
    """
    Asynchronously submits feedback to LangSmith for a specific run without blocking
    the main orchestration loop.
    """
    def _submit():
        client = Client()
        client.create_feedback(
            run_id,
            key=key,
            score=score,
            comment=comment
        )
        logger.info(f"Feedback '{key}' submitted for run {run_id} with score {score}")

    try:
        # Offload the blocking HTTP request to a background thread
        await asyncio.to_thread(_submit)
    except Exception as e:
        logger.error(f"Failed to submit feedback for run {run_id}: {e}")

