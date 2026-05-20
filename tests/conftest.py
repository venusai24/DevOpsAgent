"""
tests/conftest.py

Pytest fixtures for the AIRS test suite.
Spins up the mock FastAPI server in a background process so the tools can hit it,
and provides an in-memory LangGraph checkpointer for the graph fixture.
"""

import os
import subprocess
import time
from typing import AsyncGenerator

import httpx
import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.orchestrator import _build_compiled_graph

load_dotenv()

@pytest.fixture(scope="session", autouse=True)
def mock_enterprise_api():
    """Start the mock FastAPI server in a background process for the test session."""
    port = "8001"
    os.environ["MOCK_API_BASE_URL"] = f"http://127.0.0.1:{port}"
    # Use in-memory SQLite for test checkpointers
    os.environ["CHECKPOINT_DB_PATH"] = ":memory:"
    
    # Start uvicorn
    process = subprocess.Popen(
        ["uvicorn", "mock_enterprise.api:app", "--host", "127.0.0.1", "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Wait for liveness probe
    for _ in range(50):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        process.terminate()
        raise RuntimeError(f"Mock enterprise API failed to start on port {port}.")
        
    yield
    
    # Teardown
    process.terminate()
    process.wait()


@pytest.fixture
async def graph() -> AsyncGenerator:
    """Yield a compiled graph instance backed by an in-memory checkpointer."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield _build_compiled_graph(checkpointer)
