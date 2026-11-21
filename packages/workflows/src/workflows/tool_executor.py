"""Unified ToolExecutor: single interface for direct/HTTP MCP transport + contract deadlines."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.auth import mint_mcp_token
from core.config import Settings
from core.models import RunContext
from core.tool_contract import TOOL_CONTRACTS, invoke_with_contract
from mcp_servers.domain_store import DomainStore
from workflows.mcp_client import call_mcp_tool


@dataclass(frozen=True)
class WriteSpec:
    """Resolved write operation carrying its tool ID, endpoint, args, and direct fallback."""

    tool_id: str
    mcp_url: str
    mcp_args: dict[str, Any]
    direct_fn: Callable[[], Awaitable[dict[str, Any]]]
    reconcile_mcp_url: str


class ToolExecutor:
    """Routes read, write, and reconciliation calls over HTTP MCP or direct DomainStore."""

    def __init__(
        self,
        *,
        settings: Settings,
        domain_store: DomainStore,
        use_mcp_http: bool = False,
    ) -> None:
        self.settings = settings
        self.domain = domain_store
        self.use_mcp_http = use_mcp_http

    async def invoke_read(
        self,
        ctx: RunContext,
        *,
        tool_id: str,
        mcp_url: str,
        mcp_args: dict[str, Any],
        direct_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        contract = TOOL_CONTRACTS[tool_id]

        async def _call() -> Any:
            if self.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    secret=self.settings.mcp_token_secret,
                )
                return await call_mcp_tool(
                    url=mcp_url,
                    bearer_token=tok,
                    tool_name=tool_id,
                    arguments=mcp_args,
                )
            return await direct_fn()

        return await invoke_with_contract(contract, _call)

    async def invoke_write(
        self,
        ctx: RunContext,
        *,
        spec: WriteSpec,
        approved_by: str,
        operation_id: str,
        simulate_timeout: bool = False,
    ) -> dict[str, Any]:
        contract = TOOL_CONTRACTS[spec.tool_id]

        async def _call() -> dict[str, Any]:
            if simulate_timeout:
                raise TimeoutError(f"Simulated downstream timeout for {spec.tool_id}")
            if self.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    approved_by=approved_by,
                    operation_id=operation_id,
                    secret=self.settings.mcp_token_secret,
                )
                return await call_mcp_tool(
                    url=spec.mcp_url,
                    bearer_token=tok,
                    tool_name=spec.tool_id,
                    arguments=spec.mcp_args,
                )
            return await spec.direct_fn()

        return await invoke_with_contract(contract, _call)

    async def reconcile_operation(
        self,
        ctx: RunContext,
        *,
        mcp_url: str,
        operation_id: str,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        contract = TOOL_CONTRACTS["get_operation_status"].model_copy(
            update={"timeout_s": self.settings.tool_timeout_s}
        )

        async def _call() -> dict[str, Any]:
            if self.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    approved_by=approved_by,
                    operation_id=operation_id,
                    secret=self.settings.mcp_token_secret,
                )
                return await call_mcp_tool(
                    url=mcp_url,
                    bearer_token=tok,
                    tool_name="get_operation_status",
                    arguments={"operation_id": operation_id},
                )
            return await self.domain.get_operation_status(
                tenant_id=ctx.tenant_id, operation_id=operation_id
            )

        try:
            return await invoke_with_contract(contract, _call)
        except Exception as exc:
            return {
                "operation_id": operation_id,
                "status": "outcome_unknown",
                "error": str(exc),
            }
