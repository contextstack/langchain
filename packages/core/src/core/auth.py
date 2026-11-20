"""JWT minting and verification for API callers and internal MCP calls (spec §4, §6)."""

from __future__ import annotations

import time
from typing import Any

import jwt
from pydantic import BaseModel

from core.models import CallerIdentity

JWT_ALGORITHM = "HS256"
MCP_AUDIENCE = "mcp"
API_AUDIENCE = "ops-api"


class AuthError(ValueError):
    """Raised when a token is missing, malformed, expired, or has an invalid signature."""


class MCPTokenClaims(BaseModel, frozen=True):
    """Verified claims carried by a short-lived internal MCP bearer token."""

    tenant_id: str
    requester_id: str
    run_id: str
    approved_by: str | None = None
    operation_id: str | None = None


def mint_user_jwt(
    identity: CallerIdentity,
    *,
    secret: str,
    ttl_s: int = 3600,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "aud": API_AUDIENCE,
        "sub": identity.actor_id,
        "tenant_id": identity.tenant_id,
        "roles": sorted(identity.roles),
        "iat": now,
        "exp": now + ttl_s,
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_user_jwt(token: str, *, secret: str) -> CallerIdentity:
    if not token:
        raise AuthError("Missing bearer token")
    try:
        decoded = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            audience=API_AUDIENCE,
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"Invalid user token: {exc}") from exc

    tenant_id = decoded.get("tenant_id")
    actor_id = decoded.get("sub")
    roles = decoded.get("roles")
    if not isinstance(tenant_id, str) or not isinstance(actor_id, str) or not isinstance(
        roles, list
    ):
        raise AuthError("Malformed user token claims")

    return CallerIdentity(
        tenant_id=tenant_id,
        actor_id=actor_id,
        roles=frozenset(str(r) for r in roles),
    )


def mint_mcp_token(
    *,
    tenant_id: str,
    requester_id: str,
    run_id: str,
    secret: str,
    approved_by: str | None = None,
    operation_id: str | None = None,
    ttl_s: int = 480,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "aud": MCP_AUDIENCE,
        "tenant_id": tenant_id,
        "requester_id": requester_id,
        "run_id": run_id,
        "iat": now,
        "exp": now + ttl_s,
    }
    if approved_by is not None:
        payload["approved_by"] = approved_by
    if operation_id is not None:
        payload["operation_id"] = operation_id
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_mcp_token(token: str, *, secret: str) -> MCPTokenClaims:
    if not token:
        raise AuthError("Missing MCP bearer token")
    try:
        decoded = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            audience=MCP_AUDIENCE,
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"Invalid MCP token: {exc}") from exc

    try:
        return MCPTokenClaims(
            tenant_id=decoded["tenant_id"],
            requester_id=decoded["requester_id"],
            run_id=decoded["run_id"],
            approved_by=decoded.get("approved_by"),
            operation_id=decoded.get("operation_id"),
        )
    except (KeyError, ValueError) as exc:
        raise AuthError(f"Malformed MCP token claims: {exc}") from exc
