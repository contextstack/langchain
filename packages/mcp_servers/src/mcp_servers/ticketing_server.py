"""FastMCP Mock Ticketing server (Streamable HTTP on port 8102)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from core.config import get_settings
from mcp_servers.auth_middleware import require_mcp_claims
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore


def create_ticketing_server(
    store: PostgresDomainStore | InMemoryDomainStore | None = None,
    *,
    mcp_secret: str | None = None,
) -> FastMCP:
    mcp = FastMCP("contextstack-mock-ticketing")
    domain = store or PostgresDomainStore(get_settings().database_url)

    @mcp.tool()
    async def get_ticket(ticket_id: str) -> dict[str, Any]:
        """Read an incident ticket belonging to the caller's tenant."""
        claims = require_mcp_claims(secret=mcp_secret)
        return await domain.get_ticket(tenant_id=claims.tenant_id, ticket_id=ticket_id)

    @mcp.tool()
    async def update_ticket(
        ticket_id: str,
        status: str,
        note: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Idempotently update an incident ticket (requires approved write token)."""
        claims = require_mcp_claims(secret=mcp_secret)
        if not claims.approved_by:
            raise PermissionError("Write MCP call requires an approved_by claim in token")
        return await domain.update_ticket(
            tenant_id=claims.tenant_id,
            ticket_id=ticket_id,
            status=status,
            note=note,
            operation_id=idempotency_key,
        )

    return mcp


def run() -> None:
    server = create_ticketing_server()
    server.run(transport="streamable-http", host="127.0.0.1", port=8102)
