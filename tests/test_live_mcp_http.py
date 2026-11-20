"""Live Streamable HTTP integration tests for FastMCP + `langchain.mcp.MCPAdapter` (M0 #1, AC-4)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import psycopg
import uvicorn

from api.main import create_app
from core.auth import mint_mcp_token, mint_user_jwt
from core.config import get_settings
from core.db import apply_app_migrations
from core.models import CallerIdentity
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore
from mcp_servers.infra_server import create_infra_server
from mcp_servers.ticketing_server import create_ticketing_server
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


async def test_e2e_api_worker_authenticated_mcp_http_crash_recovery_verified() -> None:
    """Full acceptance test through normal entrypoints:
    FastAPI -> ops-worker (MCP_TRANSPORT=http) -> live FastMCP Infra+Ticketing servers ->
    submit -> approve -> authenticated MCP write -> worker crash-after-write -> automatic reclaim ->
    deduplicated replay -> MCP end-state verification.
    """
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(
        settings.database_url, autocommit=True
    ) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    pg_store = PostgresDomainStore(settings.database_url)
    infra_port = _free_port()
    ticket_port = _free_port()
    infra_url = f"http://127.0.0.1:{infra_port}/mcp"
    ticket_url = f"http://127.0.0.1:{ticket_port}/mcp"

    infra_srv = uvicorn.Server(
        uvicorn.Config(
            create_infra_server(store=pg_store, mcp_secret=settings.mcp_token_secret).http_app(
                path="/mcp"
            ),
            host="127.0.0.1",
            port=infra_port,
            log_level="error",
        )
    )
    ticket_srv = uvicorn.Server(
        uvicorn.Config(
            create_ticketing_server(store=pg_store, mcp_secret=settings.mcp_token_secret).http_app(
                path="/mcp"
            ),
            host="127.0.0.1",
            port=ticket_port,
            log_level="error",
        )
    )
    infra_task = asyncio.create_task(infra_srv.serve())
    ticket_task = asyncio.create_task(ticket_srv.serve())

    base_env = os.environ.copy()
    base_env.update(
        {
            "MCP_TRANSPORT": "http",
            "MCP_INFRA_URL": infra_url,
            "MCP_TICKETING_URL": ticket_url,
            "WORKER_LEASE_S": "2",
            "WORKER_HEARTBEAT_S": "1",
            "MODEL_MODE": "mock",
            "TELEMETRY_BACKEND": "none",
        }
    )

    def _spawn(extra: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        env = base_env.copy()
        if extra:
            env.update(extra)
        return subprocess.Popen(
            [sys.executable, "-c", "from worker.main import run; run()"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _stop(p: subprocess.Popen[bytes] | None) -> None:
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    alice_hdr = {
        "Authorization": f"Bearer {
            mint_user_jwt(
                CallerIdentity(
                    tenant_id='tenant-alpha', actor_id='alice', roles=frozenset({'engineer'})
                ),
                secret=settings.jwt_signing_secret,
            )
        }"
    }
    bob_hdr = {
        "Authorization": f"Bearer {
            mint_user_jwt(
                CallerIdentity(
                    tenant_id='tenant-alpha', actor_id='bob', roles=frozenset({'ops-approver'})
                ),
                secret=settings.jwt_signing_secret,
            )
        }"
    }

    w1: subprocess.Popen[bytes] | None = None
    w_crash: subprocess.Popen[bytes] | None = None
    w2: subprocess.Popen[bytes] | None = None
    try:
        while not (infra_srv.started and ticket_srv.started):
            await asyncio.sleep(0.05)

        w1 = _spawn()
        app = create_app()
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice_hdr)).json()[
                "thread_id"
            ]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice_hdr,
                )
            ).json()["run_id"]

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                st = (await client.get(f"/v1/runs/{r_id}", headers=alice_hdr)).json()["status"]
                if st == "awaiting_approval":
                    break
                await asyncio.sleep(0.2)
            assert st == "awaiting_approval"
            _stop(w1)

            apr_id = (await client.get("/v1/approvals", headers=bob_hdr)).json()[0]["approval_id"]

            # Approve -> Worker crashes AFTER authenticated MCP HTTP write commits
            w_crash = _spawn({"FAULT_INJECT": "after_write"})
            dec = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve", "comment": "approved over MCP HTTP"},
                headers=bob_hdr,
            )
            assert dec.status_code == 202
            resume_run_id = dec.json()["run_id"]

            await asyncio.to_thread(w_crash.wait, 10)
            assert w_crash.returncode == 137

            # Healthy worker reclaims -> replays over MCP HTTP -> deduplicated + verified
            w2 = _spawn()
            deadline = time.monotonic() + 15.0
            final_run: dict[str, object] = {}
            while time.monotonic() < deadline:
                final_run = (
                    await client.get(f"/v1/runs/{resume_run_id}", headers=alice_hdr)
                ).json()
                if final_run["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.2)
            assert final_run["status"] == "succeeded"
            assert final_run["thread_status"] == "completed"

            # Verify DB end-state + single mock_operations row + audit trail
            svc = await pg_store.get_service_status(
                tenant_id="tenant-alpha", service_name="checkout-api"
            )
            assert svc["current_version"] == "v2.4.0"
            assert svc["status"] == "healthy"
            async with (
                await psycopg.AsyncConnection.connect(settings.database_url) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute("SELECT COUNT(*) FROM mock_operations")
                count_row = await cur.fetchone()
                assert count_row is not None and count_row[0] == 1

            audits = (await client.get(f"/v1/audit?thread_id={t_id}", headers=alice_hdr)).json()
            assert any(
                a["event_type"] == "tool_write"
                and a["requested_by"] == "alice"
                and a["approved_by"] == "bob"
                for a in audits
            )

            # Also verify AC-16 cancel-after-write reconciliation routes over MCP HTTP
            t2_id = (await client.post("/v1/threads", json={}, headers=alice_hdr)).json()[
                "thread_id"
            ]
            r2_id = (
                await client.post(
                    f"/v1/threads/{t2_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice_hdr,
                )
            ).json()["run_id"]
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                st2 = (await client.get(f"/v1/runs/{r2_id}", headers=alice_hdr)).json()["status"]
                if st2 == "awaiting_approval":
                    break
                await asyncio.sleep(0.2)
            assert st2 == "awaiting_approval"
            _stop(w2)
            w2 = None

            apr2_id = (await client.get("/v1/approvals", headers=bob_hdr)).json()[0]["approval_id"]
            w_crash2 = _spawn({"FAULT_INJECT": "after_write"})
            try:
                dec2 = await client.post(
                    f"/v1/approvals/{apr2_id}/decision",
                    json={"decision": "approve"},
                    headers=bob_hdr,
                )
                assert dec2.status_code == 202
                res2_id = dec2.json()["run_id"]
                await asyncio.to_thread(w_crash2.wait, 10)
                assert w_crash2.returncode == 137

                cancel2 = await client.post(f"/v1/runs/{res2_id}/cancel", headers=alice_hdr)
                assert cancel2.status_code == 200
                assert cancel2.json()["cancel_requested"] is True

                w2 = _spawn()
                deadline = time.monotonic() + 15.0
                rec_run: dict[str, object] = {}
                while time.monotonic() < deadline:
                    rec_run = (
                        await client.get(f"/v1/runs/{res2_id}", headers=alice_hdr)
                    ).json()
                    if rec_run["status"] in {"succeeded", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.2)
                assert rec_run["status"] == "succeeded"
                audits2 = (
                    await client.get(f"/v1/audit?thread_id={t2_id}", headers=alice_hdr)
                ).json()
                assert any(
                    a["event_type"] == "reconciliation" and a["decision"] == "succeeded"
                    for a in audits2
                )
            finally:
                _stop(w_crash2)
    finally:
        _stop(w1)
        _stop(w_crash)
        _stop(w2)
        infra_srv.should_exit = True
        ticket_srv.should_exit = True
        await asyncio.gather(infra_task, ticket_task)
