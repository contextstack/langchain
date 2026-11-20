"""FastMCP Mock Infrastructure server (Streamable HTTP on port 8101)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from core.config import get_settings
from mcp_servers.auth_middleware import require_mcp_claims
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore


def create_infra_server(
    store: PostgresDomainStore | InMemoryDomainStore | None = None,
    *,
    mcp_secret: str | None = None,
) -> FastMCP:
    mcp = FastMCP("contextstack-mock-infra")
    domain = store or PostgresDomainStore(get_settings().database_url)

    @mcp.tool()
    async def get_service_status(service_name: str) -> dict[str, Any]:
        """Get the current version and health status of a service in the caller's tenant."""
        claims = require_mcp_claims(secret=mcp_secret)
        return await domain.get_service_status(
            tenant_id=claims.tenant_id,
            service_name=service_name,
        )

    @mcp.tool()
    async def list_recent_deploys(service_name: str) -> list[dict[str, Any]]:
        """List recent deploys for a service in the caller's tenant."""
        claims = require_mcp_claims(secret=mcp_secret)
        return await domain.list_recent_deploys(
            tenant_id=claims.tenant_id,
            service_name=service_name,
        )

    @mcp.tool()
    async def get_operation_status(operation_id: str) -> dict[str, Any]:
        """Reconcile the committed outcome of an external operation by idempotency key."""
        claims = require_mcp_claims(secret=mcp_secret)
        return await domain.get_operation_status(
            tenant_id=claims.tenant_id,
            operation_id=operation_id,
        )

    @mcp.tool()
    async def execute_service_change(
        service_name: str,
        action: str,
        idempotency_key: str,
        target_version: str | None = None,
    ) -> dict[str, Any]:
        """Execute an idempotent infrastructure change (requires approved write token)."""
        claims = require_mcp_claims(secret=mcp_secret)
        if not claims.approved_by:
            raise PermissionError("Write MCP call requires an approved_by claim in token")
        return await domain.execute_service_change(
            tenant_id=claims.tenant_id,
            service_name=service_name,
            action=action,
            operation_id=idempotency_key,
            target_version=target_version,
        )

    return mcp


def run() -> None:
    server = create_infra_server()
    server.run(transport="streamable-http", host="127.0.0.1", port=8101)
