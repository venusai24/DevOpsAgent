"""Observability configuration for LangSmith integration."""

import logging
import os

logger = logging.getLogger(__name__)

def init_observability(project_name: str = "DevOpsAgent-Incidents", run_id: str = None) -> None:
    """
    Initializes LangSmith tracing by setting the required environment variables.
    This ensures traces are grouped under the correct project and easily searchable.
    
    Args:
        project_name (str): The LangSmith project to record traces to.
        run_id (str): An optional global run ID (e.g., incident ID) to add as a default tag.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Adapt to custom LANGSMITH_* variables from .env if present
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    if os.environ.get("LANGSMITH_TRACING"):
        # The correct variable for enabling tracing is LANGCHAIN_TRACING_V2
        os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ["LANGSMITH_TRACING"])
    if os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
    if os.environ.get("LANGSMITH_PROJECT"):
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])

    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
        
        # We optionally use tags for filtering in the UI
        tags = []
        if run_id:
            tags.append(f"incident:{run_id}")
            
        if tags:
            os.environ["LANGCHAIN_TAGS"] = ",".join(tags)
            
        logger.info(f"LangSmith observability enabled. Project: {os.environ.get('LANGCHAIN_PROJECT')}")
    else:
        logger.warning("LANGCHAIN_API_KEY or LANGSMITH_API_KEY is not set. LangSmith tracing is disabled.")
