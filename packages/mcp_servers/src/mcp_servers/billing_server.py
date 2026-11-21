"""FastMCP Mock Billing server (Streamable HTTP on port 8103)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from core.config import get_settings
from mcp_servers.auth_middleware import require_mcp_claims
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore


def create_billing_server(
    store: PostgresDomainStore | InMemoryDomainStore | None = None,
    *,
    mcp_secret: str | None = None,
) -> FastMCP:
    mcp = FastMCP("contextstack-mock-billing")
    domain = store or PostgresDomainStore(get_settings().database_url)

    @mcp.tool()
    async def get_account(account_id: str) -> dict[str, Any]:
        """Inspect a customer billing account in the caller's tenant."""
        claims = require_mcp_claims(secret=mcp_secret)
        return await domain.get_account(
            tenant_id=claims.tenant_id,
            account_id=account_id,
        )

    @mcp.tool()
    async def get_operation_status(operation_id: str) -> dict[str, Any]:
        """Reconcile the committed outcome of a billing operation by idempotency key."""
        claims = require_mcp_claims(secret=mcp_secret)
        return await domain.get_operation_status(
            tenant_id=claims.tenant_id,
            operation_id=operation_id,
        )

    @mcp.tool()
    async def issue_refund(
        account_id: str,
        amount_cents: int,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Issue an idempotent customer refund (requires approved write token)."""
        claims = require_mcp_claims(secret=mcp_secret)
        if not claims.approved_by:
            raise PermissionError("Write MCP call requires an approved_by claim in token")
        return await domain.issue_refund(
            tenant_id=claims.tenant_id,
            account_id=account_id,
            amount_cents=amount_cents,
            reason=reason,
            operation_id=idempotency_key,
        )

    return mcp


def run() -> None:
    server = create_billing_server()
    server.run(transport="streamable-http", host="127.0.0.1", port=8103)
