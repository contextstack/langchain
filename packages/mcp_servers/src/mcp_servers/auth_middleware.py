"""MCP Bearer JWT authentication helper (spec §6).

MCP servers never accept X-Tenant-Id headers or tenant_id tool parameters.
Tenant and actor identity are derived exclusively from verified internal JWTs.
"""

from __future__ import annotations

from fastmcp.server.dependencies import get_http_headers

from core.auth import AuthError, MCPTokenClaims, verify_mcp_token
from core.config import get_settings


def extract_bearer_token(headers: dict[str, str]) -> str:
    auth_header = headers.get("authorization") or headers.get("Authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        raise AuthError("Missing Bearer authorization header")
    token = auth_header[7:].strip()
    if not token:
        raise AuthError("Empty Bearer token")
    return token


def require_mcp_claims(*, secret: str | None = None) -> MCPTokenClaims:
    """Verify the current FastMCP HTTP request's Bearer token and return claims."""
    headers = get_http_headers(include_all=True)
    token = extract_bearer_token(headers)
    signing_secret = secret or get_settings().mcp_token_secret
    return verify_mcp_token(token, secret=signing_secret)


def require_approved_write_claims(
    *,
    idempotency_key: str,
    secret: str | None = None,
) -> MCPTokenClaims:
    """Verify an MCP write request carries approved_by and a matching signed operation_id."""
    claims = require_mcp_claims(secret=secret)
    if not claims.approved_by:
        raise PermissionError("Write MCP call requires an approved_by claim in token")
    if not claims.operation_id or claims.operation_id != idempotency_key:
        raise PermissionError(
            "Write MCP call idempotency_key must match signed operation_id claim"
        )
    return claims
