"""
airs_v2/context/client.py
==========================

ContextMCPClient — Async stdio Client for the Context MCP Server
----------------------------------------------------------------

A thin async helper that wraps ``mcp.ClientSession`` + ``stdio_client`` to
provide a clean Pythonic API for calling the Context MCP server from the
reasoning engine and integration tests.

Usage
-----
::

    from airs_v2.context.client import ContextMCPClient

    async with ContextMCPClient() as client:
        tools = await client.list_tools()
        blast = await client.call("get_blast_radius", service="payments-db")

The client spawns the MCP server as a subprocess using::

    python -m airs_v2.context.mcp_server

Design note (anyio compatibility)
----------------------------------
``mcp.client.stdio.stdio_client`` is an async generator that runs inside an
``anyio`` cancel scope.  anyio requires that cancel scopes are entered and
exited within the same task.  To honour this constraint the entire lifecycle
(open transport → create session → initialize → yield → close) is kept inside
a single async-generator body decorated with ``@asynccontextmanager``.

The public ``ContextMCPClient`` class acts as a thin wrapper so callers use
the familiar ``async with`` syntax without caring about the internal generator.

Error handling
--------------
All ``call()`` responses are JSON-decoded automatically.  If the server
returns a ``{"error": "..."}`` envelope, ``CallError`` is raised with the
full error dict so callers can distinguish topology errors from transport errors.
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class CallError(Exception):
    """Raised when the MCP server returns an error envelope."""
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


class _BoundClient:
    """
    Holds a live ``ClientSession`` and exposes ergonomic tool-calling methods.
    Instances are yielded by the ``ContextMCPClient`` context manager —
    they are never constructed directly.
    """

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def list_tools(self) -> list[str]:
        """Return the names of all tools registered on the server."""
        result = await self._session.list_tools()
        return [t.name for t in result.tools]

    async def call(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Call a named MCP tool and return the decoded JSON response.

        Parameters
        ----------
        tool_name:
            Name of the tool to call (e.g. ``"get_blast_radius"``).
        **kwargs:
            Arguments forwarded to the tool.

        Returns
        -------
        Any
            The JSON-decoded response body (dict, list, or scalar).

        Raises
        ------
        CallError
            If the server returns a response containing an ``"error"`` key.
        """
        result = await self._session.call_tool(tool_name, arguments=kwargs)
        raw = result.content[0].text if result.content else "{}"
        decoded = json.loads(raw)
        if isinstance(decoded, dict) and "error" in decoded:
            raise CallError(decoded["error"], decoded)
        return decoded

    async def call_raw(self, tool_name: str, **kwargs: Any) -> str:
        """Return the raw text response without JSON decoding or error checking."""
        result = await self._session.call_tool(tool_name, arguments=kwargs)
        return result.content[0].text if result.content else ""


class ContextMCPClient:
    """
    Async context manager that opens a stdio connection to the
    ContextMCPServer subprocess and yields a ``_BoundClient``.

    The entire ``anyio`` cancel-scope lifecycle (transport open → session
    initialize → yield → close) is kept inside a single async generator so
    anyio does not raise a cross-task cancel-scope error.

    Parameters
    ----------
    python_executable:
        Path to the Python interpreter to use for the server subprocess.
        Defaults to ``sys.executable`` (same interpreter as the caller).
    extra_env:
        Optional dict of extra environment variables forwarded to the
        server process (e.g., to override fixture paths in tests).
    """

    def __init__(
        self,
        python_executable: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        self._python = python_executable or sys.executable
        self._extra_env = extra_env or None
        self._cm = None
        self._client: Optional[_BoundClient] = None

    async def __aenter__(self) -> _BoundClient:
        self._cm = self._open()
        self._client = await self._cm.__aenter__()
        return self._client

    async def __aexit__(self, *exc_info) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(*exc_info)

    @asynccontextmanager
    async def _open(self) -> AsyncGenerator[_BoundClient, None]:
        """
        Internal async context manager that keeps the entire anyio cancel
        scope inside one async-generator frame.
        """
        params = StdioServerParameters(
            command=self._python,
            args=["-m", "airs_v2.context.mcp_server"],
            env=self._extra_env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield _BoundClient(session)
