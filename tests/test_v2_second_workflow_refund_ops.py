"""v2.4 boundary-first test: second domain workflow (`refund_ops`) on the shared substrate.

Proves "swap the tools, change the factory" over real TCP sockets:
  - Zero changes to `services/api`: `POST /v1/threads` with `workflow_name="refund_ops"`
  - Over-ceiling refund ($10,000 > $5,000) denied by PolicyEngine with zero mutations
  - Valid $150 refund pauses at `awaiting_approval`, refuses self-approval, enforces tenant
    isolation, approves via `bob`, executes over authenticated FastMCP Billing HTTP, survives
    `FAULT_INJECT=after_write` (`os._exit(137)`) + automatic worker lease reclaim with
    exactly-once balance deduction (`100_000 -> 85_000`), and passes `verify_end_state`
  - Fail-closed Postgres RLS on `mock_accounts` verified via raw SQL without `WHERE`
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys

import httpx
import psycopg
import uvicorn

from api.main import create_app
from core.auth import mint_mcp_token
from core.config import get_settings
from core.db import apply_app_migrations, set_tenant_context
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.billing_server import create_billing_server
from mcp_servers.domain_store import PostgresDomainStore
from workflows.mcp_client import call_mcp_tool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stop(p: subprocess.Popen[bytes] | None) -> None:
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


async def _poll_run_status(
    client: httpx.AsyncClient,
    api_base: str,
    headers: dict[str, str],
    run_id: str,
    target_statuses: set[str],
    *,
    timeout_s: float = 25.0,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, object] = {}
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"{api_base}/v1/runs/{run_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in target_statuses:
            return last
        await asyncio.sleep(0.25)
    raise AssertionError(f"Run {run_id} did not reach {target_statuses}; last={last}")


async def test_refund_ops_second_workflow_end_to_end_with_crash_recovery_and_rls() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        seeded = await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())
    assert seeded["accounts"] >= 2

    # 1. Verify fail-closed Postgres RLS on mock_accounts (raw SELECT, no WHERE)
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        async with conn.transaction(), conn.cursor() as cur:
            await cur.execute("SELECT account_id FROM mock_accounts")
            assert await cur.fetchall() == []
        async with conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, "tenant-alpha")
            await cur.execute("SELECT account_id FROM mock_accounts")
            assert {r[0] for r in await cur.fetchall()} == {"ACC-1001"}
        async with conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, "tenant-beta")
            await cur.execute("SELECT account_id FROM mock_accounts")
            assert {r[0] for r in await cur.fetchall()} == {"ACC-2001"}

    # 2. Boot live FastMCP Billing server + live FastAPI server over real TCP sockets
    pg_store = PostgresDomainStore(settings.database_url)
    billing_port = _free_port()
    api_port = _free_port()
    billing_url = f"http://127.0.0.1:{billing_port}/mcp"
    api_base = f"http://127.0.0.1:{api_port}"

    billing_srv = uvicorn.Server(
        uvicorn.Config(
            create_billing_server(store=pg_store, mcp_secret=settings.mcp_token_secret).http_app(
                path="/mcp"
            ),
            host="127.0.0.1",
            port=billing_port,
            log_level="error",
        )
    )
    api_srv = uvicorn.Server(
        uvicorn.Config(
            create_app(settings),
            host="127.0.0.1",
            port=api_port,
            log_level="error",
        )
    )
    billing_task = asyncio.create_task(billing_srv.serve())
    api_task = asyncio.create_task(api_srv.serve())

    base_worker_env = {
        **os.environ,
        "MODEL_MODE": "mock",
        "MCP_TRANSPORT": "http",
        "MCP_BILLING_URL": billing_url,
        "WORKER_LEASE_S": "2",
        "WORKER_HEARTBEAT_S": "1",
    }
    worker_crash: subprocess.Popen[bytes] | None = None
    worker_healthy: subprocess.Popen[bytes] | None = None

    try:
        while not (billing_srv.started and api_srv.started):
            await asyncio.sleep(0.05)

        worker_healthy = subprocess.Popen(
            [sys.executable, "-c", "from worker.main import run; run()"],
            env=base_worker_env,
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            alice_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "alice"})
            ).json()["token"]
            bob_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "bob"})
            ).json()["token"]
            carol_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "carol"})
            ).json()["token"]

            alice_h = {"Authorization": f"Bearer {alice_tok}"}
            bob_h = {"Authorization": f"Bearer {bob_tok}"}
            carol_h = {"Authorization": f"Bearer {carol_tok}"}

            # 3. Policy ceiling denial: $10,000 refund (> $5,000 max) -> denied, 0 mutations
            t_deny = (
                await client.post(
                    f"{api_base}/v1/threads",
                    headers=alice_h,
                    json={"workflow_name": "refund_ops"},
                )
            ).json()["thread_id"]
            r_deny = (
                await client.post(
                    f"{api_base}/v1/threads/{t_deny}/runs",
                    headers=alice_h,
                    json={"request_text": "Issue $10,000 refund to ACC-1001 for outage"},
                )
            ).json()["run_id"]
            deny_state = await _poll_run_status(
                client, api_base, alice_h, r_deny, {"succeeded", "failed"}
            )
            assert deny_state["error_detail"] == "denied"

            acct_after_deny = await call_mcp_tool(
                url=billing_url,
                bearer_token=mint_mcp_token(
                    tenant_id="tenant-alpha",
                    requester_id="alice",
                    run_id=r_deny,
                    secret=settings.mcp_token_secret,
                ),
                tool_name="get_account",
                arguments={"account_id": "ACC-1001"},
            )
            assert acct_after_deny["balance_cents"] == 100_000

            # 4. Valid $150 refund -> pauses at awaiting_approval
            t_ok = (
                await client.post(
                    f"{api_base}/v1/threads",
                    headers=alice_h,
                    json={"workflow_name": "refund_ops"},
                )
            ).json()["thread_id"]
            r_ok = (
                await client.post(
                    f"{api_base}/v1/threads/{t_ok}/runs",
                    headers=alice_h,
                    json={"request_text": "Issue $150 refund to ACC-1001 for SLA breach"},
                )
            ).json()["run_id"]
            await _poll_run_status(client, api_base, alice_h, r_ok, {"awaiting_approval"})

            # Tenant isolation + self-approval refusal
            carol_resp = await client.get(f"{api_base}/v1/threads/{t_ok}", headers=carol_h)
            assert carol_resp.status_code == 404
            approvals = (await client.get(f"{api_base}/v1/approvals", headers=bob_h)).json()
            matching = [a for a in approvals if a["thread_id"] == t_ok]
            assert len(matching) == 1
            approval_id = matching[0]["approval_id"]
            operation_id = matching[0]["operation_id"]

            self_resp = await client.post(
                f"{api_base}/v1/approvals/{approval_id}/decision",
                headers=alice_h,
                json={"decision": "approve"},
            )
            assert self_resp.status_code == 403

            # Swap in a crashing worker (FAULT_INJECT=after_write) BEFORE Bob approves
            _stop(worker_healthy)
            worker_healthy = None
            worker_crash = subprocess.Popen(
                [sys.executable, "-c", "from worker.main import run; run()"],
                env={**base_worker_env, "FAULT_INJECT": "after_write"},
            )

            dec_resp = await client.post(
                f"{api_base}/v1/approvals/{approval_id}/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
            assert dec_resp.status_code == 202
            resume_run_id = dec_resp.json()["run_id"]

            # Wait for Worker #1 to crash right after committing the MCP HTTP refund
            rc = await asyncio.to_thread(worker_crash.wait, 20)
            assert rc == 137

            # Start healthy Worker #2 to reclaim the expired lease and finish verification
            worker_healthy = subprocess.Popen(
                [sys.executable, "-c", "from worker.main import run; run()"],
                env=base_worker_env,
            )
            final_run = await _poll_run_status(
                client, api_base, bob_h, resume_run_id, {"succeeded", "failed"}
            )
            assert final_run["status"] == "succeeded", final_run
            assert final_run["attempt"] >= 2

            # Verify exactly-once balance deduction over authenticated MCP HTTP
            acct_after_ok = await call_mcp_tool(
                url=billing_url,
                bearer_token=mint_mcp_token(
                    tenant_id="tenant-alpha",
                    requester_id="bob",
                    run_id=resume_run_id,
                    secret=settings.mcp_token_secret,
                ),
                tool_name="get_account",
                arguments={"account_id": "ACC-1001"},
            )
            assert acct_after_ok["balance_cents"] == 85_000

            # Verify mock_operations count == 1 and audit_log captures separation of duties
            async with (
                await psycopg.AsyncConnection.connect(settings.database_url) as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                    await set_tenant_context(conn, "tenant-alpha")
                    await cur.execute(
                        "SELECT count(*) FROM mock_operations WHERE operation_id = %s",
                        (operation_id,),
                    )
                    assert (await cur.fetchone())[0] == 1
                    await cur.execute(
                        """
                        SELECT requested_by, approved_by, tool_id, decision
                          FROM audit_log
                         WHERE thread_id = %s AND event_type = 'tool_write'
                        """,
                        (t_ok,),
                    )
                    writes = await cur.fetchall()
                    assert writes, "Expected tool_write audit row for refund_ops"
                    assert any(
                        r[0] == "alice" and r[1] == "bob" and r[2] == "issue_refund"
                        for r in writes
                    )
    finally:
        _stop(worker_crash)
        _stop(worker_healthy)
        billing_srv.should_exit = True
        api_srv.should_exit = True
        await asyncio.gather(billing_task, api_task, return_exceptions=True)
