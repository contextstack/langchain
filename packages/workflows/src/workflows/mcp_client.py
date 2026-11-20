"""Authenticated LangChain MCP client helper using native `langchain.mcp.MCPAdapter` (spec §6)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Client as FastMCPClient
from langchain.mcp import MCPAdapter


class MCPToolCallError(RuntimeError):
    """Raised when an MCP tool call returns an error response."""


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


def _parse_mcp_result(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        text = (
            str(first["text"])
            if isinstance(first, dict) and "text" in first
            else str(first)
        )
    else:
        text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MCPToolCallError(text) from exc


async def call_mcp_tool(
    *,
    url: str,
    bearer_token: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Invoke a named MCP tool over Streamable HTTP with a signed Bearer JWT."""
    async with authenticated_mcp_adapter(url=url, bearer_token=bearer_token) as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
        if tool_name not in tools:
            raise MCPToolCallError(f"MCP tool '{tool_name}' not found at {url}")
        raw = await tools[tool_name].ainvoke(arguments)
        return _parse_mcp_result(raw)
