"""v2.7 boundary-first test: multi-worker concurrency & contention over live TCP.

Boots 3 simultaneous `ops-worker` processes against live FastAPI + FastMCP Infra +
FastMCP Billing servers (`MCP_TRANSPORT=http`) and live PostgreSQL/Redis, then submits
and approves a concurrent mixed batch of `refund_ops` and `incident_desk` runs:
  - 3 concurrent $100 refunds on ACC-1001 (100_000 -> 70_000 cents, zero lost updates)
  - 1 `incident_desk` rollback on checkout-api
  - 1 `incident_desk` status-only run
  - 1 same-thread `enqueue` pair proving per-thread serialization under 3 parallel workers
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
from mcp_servers.infra_server import create_infra_server
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


async def _poll_run(
    client: httpx.AsyncClient,
    api_base: str,
    headers: dict[str, str],
    run_id: str,
    target_statuses: set[str],
    *,
    timeout_s: float = 30.0,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, object] = {}
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"{api_base}/v1/runs/{run_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in target_statuses:
            return last
        await asyncio.sleep(0.2)
    raise AssertionError(f"Run {run_id} did not reach {target_statuses}; last={last}")


async def test_three_parallel_workers_handle_mixed_concurrent_batch_without_duplicates() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    pg_store = PostgresDomainStore(settings.database_url)
    infra_port = _free_port()
    billing_port = _free_port()
    api_port = _free_port()
    infra_url = f"http://127.0.0.1:{infra_port}/mcp"
    billing_url = f"http://127.0.0.1:{billing_port}/mcp"
    api_base = f"http://127.0.0.1:{api_port}"

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
    tasks = [
        asyncio.create_task(infra_srv.serve()),
        asyncio.create_task(billing_srv.serve()),
        asyncio.create_task(api_srv.serve()),
    ]
    worker_env = {
        **os.environ,
        "MODEL_MODE": "mock",
        "MCP_TRANSPORT": "http",
        "MCP_INFRA_URL": infra_url,
        "MCP_BILLING_URL": billing_url,
        "WORKER_LEASE_S": "5",
        "WORKER_HEARTBEAT_S": "2",
    }
    workers: list[subprocess.Popen[bytes]] = []

    try:
        while not (infra_srv.started and billing_srv.started and api_srv.started):
            await asyncio.sleep(0.05)

        # Boot 3 real parallel worker processes
        for _ in range(3):
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-c", "from worker.main import run; run()"],
                    env=worker_env,
                )
            )

        async with httpx.AsyncClient(timeout=20.0) as client:
            alice_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "alice"})
            ).json()["token"]
            bob_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "bob"})
            ).json()["token"]
            alice_h = {"Authorization": f"Bearer {alice_tok}"}
            bob_h = {"Authorization": f"Bearer {bob_tok}"}

            async def _start(workflow_name: str, text: str) -> tuple[str, str]:
                t_id = (
                    await client.post(
                        f"{api_base}/v1/threads",
                        headers=alice_h,
                        json={"workflow_name": workflow_name},
                    )
                ).json()["thread_id"]
                r_id = (
                    await client.post(
                        f"{api_base}/v1/threads/{t_id}/runs",
                        headers=alice_h,
                        json={"request_text": text},
                    )
                ).json()["run_id"]
                return t_id, r_id

            # Submit 5 independent threads concurrently (3 refunds + 1 rollback + 1 status-only)
            started = await asyncio.gather(
                _start("refund_ops", "Issue $100 refund to ACC-1001 batch-1"),
                _start("refund_ops", "Issue $100 refund to ACC-1001 batch-2"),
                _start("refund_ops", "Issue $100 refund to ACC-1001 batch-3"),
                _start("incident_desk", "checkout-api returning 502s after v2.4.1 deploy"),
                _start("incident_desk", "status only check on checkout-api"),
            )
            hitl_runs = [r for _, r in started[:4]]
            status_only_run = started[4][1]

            # Also submit a same-thread enqueue pair to prove per-thread serialization
            enq_thread = (
                await client.post(
                    f"{api_base}/v1/threads",
                    headers=alice_h,
                    json={"workflow_name": "incident_desk"},
                )
            ).json()["thread_id"]
            enq_r1 = (
                await client.post(
                    f"{api_base}/v1/threads/{enq_thread}/runs",
                    headers=alice_h,
                    json={"request_text": "status only check 1", "multitask_strategy": "enqueue"},
                )
            ).json()["run_id"]
            enq_r2 = (
                await client.post(
                    f"{api_base}/v1/threads/{enq_thread}/runs",
                    headers=alice_h,
                    json={"request_text": "status only check 2", "multitask_strategy": "enqueue"},
                )
            ).json()["run_id"]

            # Wait for all 4 HITL runs to pause and all 3 status-only runs to succeed
            await asyncio.gather(
                *(
                    _poll_run(client, api_base, alice_h, r, {"awaiting_approval"})
                    for r in hitl_runs
                ),
                _poll_run(client, api_base, alice_h, status_only_run, {"succeeded"}),
                _poll_run(client, api_base, alice_h, enq_r1, {"succeeded"}),
                _poll_run(client, api_base, alice_h, enq_r2, {"succeeded"}),
            )

            # Fetch all 4 pending approvals and approve them CONCURRENTLY
            approvals = (await client.get(f"{api_base}/v1/approvals", headers=bob_h)).json()
            assert len(approvals) == 4

            async def _approve(approval_id: str) -> str:
                resp = await client.post(
                    f"{api_base}/v1/approvals/{approval_id}/decision",
                    headers=bob_h,
                    json={"decision": "approve"},
                )
                assert resp.status_code == 202
                return str(resp.json()["run_id"])

            resume_run_ids = await asyncio.gather(
                *(_approve(a["approval_id"]) for a in approvals)
            )
            final_states = await asyncio.gather(
                *(
                    _poll_run(client, api_base, bob_h, rid, {"succeeded", "failed"})
                    for rid in resume_run_ids
                )
            )
            for st in final_states:
                assert st["status"] == "succeeded", st
                assert st["attempt"] == 1, st

            # Verify exact balance: 100_000 - 3 * 10_000 == 70_000 (no lost updates!)
            acct = await call_mcp_tool(
                url=billing_url,
                bearer_token=mint_mcp_token(
                    tenant_id="tenant-alpha",
                    requester_id="bob",
                    run_id=resume_run_ids[0],
                    secret=settings.mcp_token_secret,
                ),
                tool_name="get_account",
                arguments={"account_id": "ACC-1001"},
            )
            assert acct["balance_cents"] == 70_000

            svc = await call_mcp_tool(
                url=infra_url,
                bearer_token=mint_mcp_token(
                    tenant_id="tenant-alpha",
                    requester_id="bob",
                    run_id=resume_run_ids[0],
                    secret=settings.mcp_token_secret,
                ),
                tool_name="get_service_status",
                arguments={"service_name": "checkout-api"},
            )
            assert svc["current_version"] == "v2.4.0"
            assert svc["status"] == "healthy"

            # Verify exactly 4 distinct operations recorded (3 refunds + 1 rollback)
            async with (
                await psycopg.AsyncConnection.connect(settings.database_url) as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                await set_tenant_context(conn, "tenant-alpha")
                await cur.execute("SELECT count(*) FROM mock_operations")
                assert (await cur.fetchone())[0] == 4
    finally:
        for w in workers:
            _stop(w)
        infra_srv.should_exit = True
        billing_srv.should_exit = True
        api_srv.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
