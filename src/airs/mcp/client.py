"""
MCP Tool Client — Module 1.6.

Direct Model Context Protocol (MCP) client for Phase 1.
Connects to MCP servers via stdio transport and invokes tools.

Phase 1: Simple direct connections with API key auth from .env.
Phase 2: Routes through API Gateway (Tyk/Kuadrant) with JWT/OIDC.
         The interface (MCPToolClient.invoke) remains identical.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class MCPToolResponse:
    """
    Normalised response from an MCP tool invocation.

    success:        Whether the tool call succeeded.
    content:        Parsed tool response (dict or list).
    raw_text:       Raw string response from the MCP server.
    raw_digest:     SHA-256 of raw_text (for provenance).
    error_code:     HTTP/MCP error code if success == False.
    error_message:  Error description if success == False.
    server_id:      Which MCP server responded.
    tool_name:      Which tool was called.
    latency_ms:     Round-trip time in milliseconds.
    """
    success: bool
    content: Any
    raw_text: str
    raw_digest: str
    error_code: Optional[int]
    error_message: Optional[str]
    server_id: str
    tool_name: str
    latency_ms: float


class MCPToolClient:
    """
    Phase 1 MCP Tool Client.

    Manages connections to MCP servers and invokes tools on behalf of
    the investigation activities. Each server is identified by a server_id
    (e.g., 'prometheus', 'loki') and connected via the URI from config.

    Thread safety: Each call creates a fresh context-managed session.
    In Phase 1, we use the MCP Python SDK's stdio client transport.
    When the MCP server URI is empty/None, the client runs in MOCK mode
    and returns a synthetic empty response (for testing without live servers).
    """

    def __init__(self, server_uris: dict[str, str | None]) -> None:
        """
        Args:
            server_uris: Mapping of server_id → URI string (or None if not configured).
                         e.g., {'prometheus': 'http://localhost:8080/mcp', 'loki': None}
        """
        self._server_uris = server_uris

    @classmethod
    def from_settings(cls) -> "MCPToolClient":
        """Create MCPToolClient from AIRSConfig settings."""
        from airs.config import settings
        return cls(
            server_uris={
                "prometheus": settings.mcp_prometheus_uri,
                "loki": settings.mcp_loki_uri,
                "opensearch": settings.mcp_opensearch_uri,
                "jaeger": settings.mcp_jaeger_uri,
                "kubectl": settings.mcp_kubectl_uri,
                "github": settings.mcp_github_uri,
            }
        )

    async def invoke(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> MCPToolResponse:
        """
        Invoke a tool on an MCP server.

        Args:
            server_id:  MCP server identifier (e.g., 'prometheus').
            tool_name:  Tool to invoke (e.g., 'execute_range_query').
            arguments:  Tool arguments dict.
            timeout:    Maximum wait time in seconds.

        Returns:
            MCPToolResponse — always returns (never raises).
            On failure: success=False with error details.
        """
        import time
        start = time.perf_counter()
        uri = self._server_uris.get(server_id)

        # ── Mock mode: no URI configured ──────────────────────────────────────
        if not uri:
            log.warning(
                "MCP server '%s' has no URI configured — returning mock response",
                server_id,
            )
            raw = json.dumps({"mock": True, "server": server_id, "tool": tool_name})
            return MCPToolResponse(
                success=True,
                content={"mock": True},
                raw_text=raw,
                raw_digest=self._digest(raw),
                error_code=None,
                error_message=None,
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # ── Real MCP invocation ───────────────────────────────────────────────
        try:
            result = await asyncio.wait_for(
                self._invoke_mcp(uri, tool_name, arguments),
                timeout=timeout,
            )
            elapsed = (time.perf_counter() - start) * 1000
            raw = json.dumps(result) if not isinstance(result, str) else result
            return MCPToolResponse(
                success=True,
                content=result,
                raw_text=raw,
                raw_digest=self._digest(raw),
                error_code=None,
                error_message=None,
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=elapsed,
            )

        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            log.error("MCP %s/%s timed out after %.0fms", server_id, tool_name, elapsed)
            return MCPToolResponse(
                success=False,
                content=None,
                raw_text="",
                raw_digest="",
                error_code=408,
                error_message=f"Timeout after {timeout}s",
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            log.error("MCP %s/%s failed: %s", server_id, tool_name, exc)
            return MCPToolResponse(
                success=False,
                content=None,
                raw_text="",
                raw_digest="",
                error_code=500,
                error_message=str(exc),
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=elapsed,
            )

    async def _invoke_mcp(
        self,
        uri: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Internal MCP invocation via the MCP Python SDK.

        Supports both HTTP+SSE transport (for remote servers) and
        stdio transport (for local MCP processes).

        Phase 1: Uses mcp.client.sse for HTTP-based MCP servers.
        """
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(uri) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                # MCP result.content is a list of content blocks
                if result.content:
                    # Extract text content from first block
                    first = result.content[0]
                    if hasattr(first, "text"):
                        try:
                            return json.loads(first.text)
                        except json.JSONDecodeError:
                            return first.text
                return {}

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
