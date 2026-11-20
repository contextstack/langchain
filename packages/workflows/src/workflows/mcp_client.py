"""Authenticated LangChain MCP client helper using native `langchain.mcp.MCPAdapter` (spec §6)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Client as FastMCPClient
from langchain.mcp import MCPAdapter


@asynccontextmanager
async def authenticated_mcp_adapter(
    *,
    url: str,
    bearer_token: str | None,
) -> AsyncIterator[MCPAdapter]:
    """Open a Streamable HTTP MCPAdapter session carrying a short-lived signed JWT."""
    client: FastMCPClient[Any] = FastMCPClient(url, auth=bearer_token)
    async with MCPAdapter(client) as adapter:
        yield adapter
