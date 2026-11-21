"""Whole-system end-to-end smoke test (v2.2).

Boots ALL real services over real TCP sockets simultaneously:
  - FastAPI (`ops-api` via uvicorn) serving REST, SSE, and the single-file Demo Web UI
  - FastMCP Infra server (Streamable HTTP, PostgresDomainStore + RLS)
  - FastMCP Ticketing server (Streamable HTTP, PostgresDomainStore + RLS)
  - Separate `ops-worker` subprocess (`MCP_TRANSPORT=http`)
  - Live PostgreSQL (+ pgvector + RLS) and Redis Streams

Exercises the exact user journey from UI page load -> dev persona tokens ->
submit incident -> real HTTP SSE stream + Last-Event-ID reconnect -> tenant isolation ->
self-approval refusal -> approval inbox -> authenticated MCP write -> SSE completion ->
MCP read-back verification -> full audit trail.
"""

from __future__ import annotations

import asyncio
import json
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
from core.db import apply_app_migrations
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import PostgresDomainStore
from mcp_servers.infra_server import create_infra_server
from mcp_servers.ticketing_server import create_ticketing_server
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


async def _consume_sse(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> list[tuple[str | None, str, dict[str, object]]]:
    events: list[tuple[str | None, str, dict[str, object]]] = []
    async with client.stream("GET", url, headers=headers, timeout=20.0) as resp:
        assert resp.status_code == 200
        ev_id: str | None = None
        ev_type = "message"
        data_lines: list[str] = []
        async for line in resp.aiter_lines():
            if line == "":
                if data_lines:
                    payload = json.loads("\n".join(data_lines))
                    events.append((ev_id, ev_type, payload))
                    if ev_type in {"done", "interrupt", "error"}:
                        break
                ev_id = None
                ev_type = "message"
                data_lines = []
            elif line.startswith("id: "):
                ev_id = line[4:].strip()
            elif line.startswith("event: "):
                ev_type = line[7:].strip()
            elif line.startswith("data: "):
                data_lines.append(line[6:])
    return events


async def test_whole_system_tcp_smoke_ui_sse_mcp_rls_and_audit() -> None:
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
    ticket_port = _free_port()
    api_port = _free_port()
    infra_url = f"http://127.0.0.1:{infra_port}/mcp"
    ticket_url = f"http://127.0.0.1:{ticket_port}/mcp"
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
    api_srv = uvicorn.Server(
        uvicorn.Config(
            create_app(),
            host="127.0.0.1",
            port=api_port,
            log_level="error",
        )
    )
    infra_task = asyncio.create_task(infra_srv.serve())
    ticket_task = asyncio.create_task(ticket_srv.serve())
    api_task = asyncio.create_task(api_srv.serve())

    worker_env = os.environ.copy()
    worker_env.update(
        {
            "MCP_TRANSPORT": "http",
            "MCP_INFRA_URL": infra_url,
            "MCP_TICKETING_URL": ticket_url,
            "WORKER_LEASE_S": "5",
            "WORKER_HEARTBEAT_S": "2",
            "MODEL_MODE": "mock",
            "TELEMETRY_BACKEND": "none",
        }
    )
    worker_proc: subprocess.Popen[bytes] | None = None

    try:
        while not (infra_srv.started and ticket_srv.started and api_srv.started):
            await asyncio.sleep(0.05)

        worker_proc = subprocess.Popen(
            [sys.executable, "-c", "from worker.main import run; run()"],
            env=worker_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        async with httpx.AsyncClient(base_url=api_base, timeout=15.0) as client:
            # 1. Static Demo Web UI & healthz over real HTTP
            assert (await client.get("/healthz")).json() == {"status": "ok"}
            ui_resp = await client.get("/")
            assert ui_resp.status_code == 200
            assert "ContextStack IT/Ops Change-Request" in ui_resp.text

            # 2. Mint persona tokens via the Demo UI's /v1/dev/token endpoint
            tokens: dict[str, dict[str, str]] = {}
            for persona in ("dana", "bob", "carol"):
                tok_resp = await client.post("/v1/dev/token", json={"persona": persona})
                assert tok_resp.status_code == 200
                tokens[persona] = {"Authorization": f"Bearer {tok_resp.json()['token']}"}

            # 3. Dana submits incident -> stream real SSE until `interrupt`
            t_resp = await client.post("/v1/threads", json={}, headers=tokens["dana"])
            assert t_resp.status_code == 201
            thread_id = t_resp.json()["thread_id"]

            r_resp = await client.post(
                f"/v1/threads/{thread_id}/runs",
                json={"request_text": "checkout-api is returning 502s after v2.4.1 deploy"},
                headers=tokens["dana"],
            )
            assert r_resp.status_code == 202
            start_run_id = r_resp.json()["run_id"]

            sse_events = await _consume_sse(
                client, f"/v1/runs/{start_run_id}/stream", headers=tokens["dana"]
            )
            ev_types = [ev[1] for ev in sse_events]
            assert "node_update" in ev_types
            assert ev_types[-1] == "interrupt"
            first_ev_id = sse_events[0][0]
            assert first_ev_id is not None

            # Verify Last-Event-ID reconnect replays subsequent events cleanly
            replayed = await _consume_sse(
                client,
                f"/v1/runs/{start_run_id}/stream",
                headers={**tokens["dana"], "Last-Event-ID": first_ev_id},
            )
            assert len(replayed) == len(sse_events) - 1
            assert replayed[-1][1] == "interrupt"

            # 4. Cross-tenant isolation (carol @ tenant-beta sees nothing)
            assert (
                await client.get(f"/v1/runs/{start_run_id}", headers=tokens["carol"])
            ).status_code == 404
            assert (await client.get("/v1/threads", headers=tokens["carol"])).json() == []
            assert (await client.get("/v1/audit", headers=tokens["carol"])).json() == []

            # 5. Separation of duties: dana cannot approve her own request
            approvals = (await client.get("/v1/approvals", headers=tokens["bob"])).json()
            assert len(approvals) == 1
            apr = approvals[0]
            assert len(apr["plan"]["citations"]) > 0
            apr_id = apr["approval_id"]

            self_resp = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve", "comment": "self-approve attempt"},
                headers=tokens["dana"],
            )
            assert self_resp.status_code == 403

            # 6. Bob approves -> stream resume run over real HTTP SSE to `done`
            dec_resp = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve", "comment": "approved in smoke test"},
                headers=tokens["bob"],
            )
            assert dec_resp.status_code == 202
            resume_run_id = dec_resp.json()["run_id"]

            resume_events = await _consume_sse(
                client, f"/v1/runs/{resume_run_id}/stream", headers=tokens["bob"]
            )
            assert resume_events[-1][1] == "done"
            assert resume_events[-1][2]["final_status"] == "completed"
            assert resume_events[-1][2]["verification_passed"] is True

            # 7. Verify live state over authenticated MCP HTTP
            read_tok = mint_mcp_token(
                tenant_id="tenant-alpha",
                requester_id="bob",
                run_id=resume_run_id,
                secret=settings.mcp_token_secret,
            )
            svc = await call_mcp_tool(
                url=infra_url,
                bearer_token=read_tok,
                tool_name="get_service_status",
                arguments={"service_name": "checkout-api"},
            )
            assert svc["current_version"] == "v2.4.0"
            assert svc["status"] == "healthy"

            # 8. Verify complete authoritative audit trail
            audits = (
                await client.get(f"/v1/audit?thread_id={thread_id}", headers=tokens["bob"])
            ).json()
            ev_names = {a["event_type"] for a in audits}
            assert {
                "authz_check",
                "approval_requested",
                "approval_refused",
                "approval_decided",
                "tool_write",
            }.issubset(ev_names)
            write_row = next(a for a in audits if a["event_type"] == "tool_write")
            assert write_row["requested_by"] == "dana"
            assert write_row["approved_by"] == "bob"
    finally:
        _stop(worker_proc)
        api_srv.should_exit = True
        infra_srv.should_exit = True
        ticket_srv.should_exit = True
        await asyncio.gather(api_task, infra_task, ticket_task)
