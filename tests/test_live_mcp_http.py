"""Live Streamable HTTP integration tests for FastMCP + `langchain.mcp.MCPAdapter` (M0 #1, AC-4)."""

from __future__ import annotations

import asyncio
import json
import socket

import uvicorn

from core.auth import mint_mcp_token
from mcp_servers.domain_store import InMemoryDomainStore
from mcp_servers.infra_server import create_infra_server
from workflows.mcp_client import authenticated_mcp_adapter

TEST_MCP_SECRET = "live-http-mcp-test-secret-key-at-least-32-bytes"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _extract_json_result(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict) and "text" in first:
            return json.loads(str(first["text"]))
        if isinstance(first, str):
            return json.loads(first)
    return json.loads(str(raw))


def _is_mcp_error(raw: object, expected_substring: str) -> bool:
    text = str(raw)
    return "error" in text.lower() and expected_substring.lower() in text.lower()


async def test_live_mcp_streamable_http_auth_isolation_and_idempotency() -> None:
    store = InMemoryDomainStore()
    server = create_infra_server(store=store, mcp_secret=TEST_MCP_SECRET)
    port = _free_port()
    mcp_url = f"http://127.0.0.1:{port}/mcp"

    asgi_app = server.http_app(path="/mcp")
    uv_config = uvicorn.Config(asgi_app, host="127.0.0.1", port=port, log_level="error")
    uv_server = uvicorn.Server(uv_config)
    serve_task = asyncio.create_task(uv_server.serve())

    try:
        while not uv_server.started:
            await asyncio.sleep(0.05)

        # 1. Valid read token for tenant-alpha
        read_tok = mint_mcp_token(
            tenant_id="tenant-alpha",
            requester_id="alice",
            run_id="run-http-1",
            secret=TEST_MCP_SECRET,
        )
        async with authenticated_mcp_adapter(url=mcp_url, bearer_token=read_tok) as adapter:
            tools = {t.name: t for t in await adapter.list_tools()}
            assert "get_service_status" in tools
            assert "execute_service_change" in tools
            # Verify no tool exposes a tenant_id argument to the LLM
            for t in tools.values():
                schema = t.args_schema if isinstance(t.args_schema, dict) else {}
                assert "tenant_id" not in schema.get("properties", {})

            status_raw = await tools["get_service_status"].ainvoke(
                {"service_name": "checkout-api"}
            )
            status = _extract_json_result(status_raw)
            assert status["tenant_id"] == "tenant-alpha"
            assert status["current_version"] == "v2.4.1"

            # Cross-tenant resource lookup fails
            cross_res = await tools["get_service_status"].ainvoke({"service_name": "billing-db"})
            assert _is_mcp_error(cross_res, "not found")

            # Write without approved_by claim refused
            unapproved_res = await tools["execute_service_change"].ainvoke(
                {
                    "service_name": "checkout-api",
                    "action": "rollback",
                    "target_version": "v2.4.0",
                    "idempotency_key": "op-unapproved",
                }
            )
            assert _is_mcp_error(unapproved_res, "approved_by")
            assert len(store.operations) == 0

        # 2. Missing token -> refused
        async with authenticated_mcp_adapter(url=mcp_url, bearer_token=None) as adapter:
            tools = {t.name: t for t in await adapter.list_tools()}
            missing_res = await tools["get_service_status"].ainvoke(
                {"service_name": "checkout-api"}
            )
            assert _is_mcp_error(missing_res, "bearer")

        # 3. Forged token (wrong secret) -> refused
        forged_tok = mint_mcp_token(
            tenant_id="tenant-alpha",
            requester_id="eve",
            run_id="run-forged",
            secret="wrong-secret-key-at-least-32-bytes-long-xyz",
        )
        async with authenticated_mcp_adapter(url=mcp_url, bearer_token=forged_tok) as adapter:
            tools = {t.name: t for t in await adapter.list_tools()}
            forged_res = await tools["get_service_status"].ainvoke(
                {"service_name": "checkout-api"}
            )
            assert _is_mcp_error(forged_res, "invalid")

        # 4. Approved write token -> idempotent commit + deduplicated replay + reconciliation
        write_tok = mint_mcp_token(
            tenant_id="tenant-alpha",
            requester_id="alice",
            run_id="run-http-1",
            approved_by="bob",
            operation_id="op-http-idem-1",
            secret=TEST_MCP_SECRET,
        )
        async with authenticated_mcp_adapter(url=mcp_url, bearer_token=write_tok) as adapter:
            tools = {t.name: t for t in await adapter.list_tools()}
            first = _extract_json_result(
                await tools["execute_service_change"].ainvoke(
                    {
                        "service_name": "checkout-api",
                        "action": "rollback",
                        "target_version": "v2.4.0",
                        "idempotency_key": "op-http-idem-1",
                    }
                )
            )
            assert first["deduplicated"] is False
            assert first["current_version"] == "v2.4.0"

            replay = _extract_json_result(
                await tools["execute_service_change"].ainvoke(
                    {
                        "service_name": "checkout-api",
                        "action": "rollback",
                        "target_version": "v2.4.0",
                        "idempotency_key": "op-http-idem-1",
                    }
                )
            )
            assert replay["deduplicated"] is True
            assert len(store.operations) == 1

            reconciled = _extract_json_result(
                await tools["get_operation_status"].ainvoke({"operation_id": "op-http-idem-1"})
            )
            assert reconciled["status"] == "succeeded"
    finally:
        uv_server.should_exit = True
        await serve_task
