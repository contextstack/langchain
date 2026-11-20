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
