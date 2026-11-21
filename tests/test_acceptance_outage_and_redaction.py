"""End-to-end acceptance tests for:
1. MCP outage during cancellation reconciliation (worker stays alive & serves queue).
2. Effective LangSmith HTTP export path PII redaction (captures actual outgoing HTTP payloads).
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import os
import socket
import subprocess
import sys
import time
from typing import Any

import httpx
import psycopg
import uvicorn
from fastapi import FastAPI, Request
from langgraph.checkpoint.memory import InMemorySaver

from api.main import create_app
from core.auth import mint_user_jwt
from core.config import Settings, get_settings
from core.db import apply_app_migrations
from core.models import CallerIdentity, RunContext
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore
from mcp_servers.infra_server import create_infra_server
from telemetry.tracing import build_langchain_callbacks, configure_tracer_provider
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph


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


# ---------------------------------------------------------------------------
# Acceptance Test 1: MCP outage during cancel reconciliation must NOT crash
# the worker or block unrelated queued work.
# ---------------------------------------------------------------------------
async def test_mcp_outage_during_cancel_reconciliation_does_not_block_worker() -> None:
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
    infra_url = f"http://127.0.0.1:{infra_port}/mcp"

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
    infra_task = asyncio.create_task(infra_srv.serve())

    base_env = os.environ.copy()
    base_env.update(
        {
            "MCP_TRANSPORT": "http",
            "MCP_INFRA_URL": infra_url,
            "MCP_TICKETING_URL": infra_url,
            "WORKER_LEASE_S": "2",
            "WORKER_HEARTBEAT_S": "1",
            "TOOL_TIMEOUT_S": "1.5",
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
    w_survivor: subprocess.Popen[bytes] | None = None
    try:
        while not infra_srv.started:
            await asyncio.sleep(0.05)

        w1 = _spawn()
        app = create_app()
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            # 1. Start and pause thread 1 for approval
            t1 = (await client.post("/v1/threads", json={}, headers=alice_hdr)).json()["thread_id"]
            r1 = (
                await client.post(
                    f"/v1/threads/{t1}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice_hdr,
                )
            ).json()["run_id"]
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if (await client.get(f"/v1/runs/{r1}", headers=alice_hdr)).json()[
                    "status"
                ] == "awaiting_approval":
                    break
                await asyncio.sleep(0.2)
            _stop(w1)

            apr_id = (await client.get("/v1/approvals", headers=bob_hdr)).json()[0]["approval_id"]

            # 2. Approve -> worker commits write and crashes via FAULT_INJECT=after_write
            w_crash = _spawn({"FAULT_INJECT": "after_write"})
            dec = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve"},
                headers=bob_hdr,
            )
            resume_run_id = dec.json()["run_id"]
            await asyncio.to_thread(w_crash.wait, 10)
            assert w_crash.returncode == 137

            # 3. Cancel the expired run AND shut down the MCP server to simulate an MCP outage!
            await client.post(f"/v1/runs/{resume_run_id}/cancel", headers=alice_hdr)
            infra_srv.should_exit = True
            await infra_task

            # 4. Enqueue an UNRELATED run on thread 2 that should still be served!
            t2 = (await client.post("/v1/threads", json={}, headers=alice_hdr)).json()["thread_id"]
            r2 = (
                await client.post(
                    f"/v1/threads/{t2}/runs",
                    json={"request_text": "status only check"},
                    headers=alice_hdr,
                )
            ).json()["run_id"]

            # 5. Start survivor worker: must mark resume_run_id as failed (outcome_unknown),
            #    stay alive, AND complete unrelated run r2!
            w_survivor = _spawn()
            deadline = time.monotonic() + 15.0
            r1_final: dict[str, Any] = {}
            r2_final: dict[str, Any] = {}
            while time.monotonic() < deadline:
                r1_final = (
                    await client.get(f"/v1/runs/{resume_run_id}", headers=alice_hdr)
                ).json()
                r2_final = (await client.get(f"/v1/runs/{r2}", headers=alice_hdr)).json()
                if r1_final["status"] in {"failed", "succeeded", "cancelled"} and r2_final[
                    "status"
                ] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.25)

            assert w_survivor.poll() is None, "Worker crashed during MCP outage sweep!"
            assert r1_final["status"] == "failed"
            assert r1_final["error_category"] == "outcome_unknown"
            assert r2_final["status"] == "succeeded"

            audits = (await client.get(f"/v1/audit?thread_id={t1}", headers=alice_hdr)).json()
            assert any(
                a["event_type"] == "reconciliation" and a["decision"] == "outcome_unknown"
                for a in audits
            )
    finally:
        _stop(w1)
        _stop(w_crash)
        _stop(w_survivor)
        if not infra_task.done():
            infra_srv.should_exit = True
            await infra_task


# ---------------------------------------------------------------------------
# Acceptance Test 2: Effective LangSmith HTTP export path PII redaction.
# Points real LangSmith tracing at a local HTTP capture server and inspects
# EVERY outgoing request payload for leaked PII.
# ---------------------------------------------------------------------------
async def test_langsmith_effective_http_export_path_masks_all_pii() -> None:
    captured_bodies: list[str] = []
    capture_app = FastAPI()

    @capture_app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH"])
    async def capture_any(full_path: str, request: Request) -> dict[str, str]:
        raw = await request.body()
        if request.headers.get("content-encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            with contextlib.suppress(Exception):
                raw = gzip.decompress(raw)
        captured_bodies.append(raw.decode("utf-8", errors="replace"))
        return {"status": "ok"}

    port = _free_port()
    cap_srv = uvicorn.Server(
        uvicorn.Config(capture_app, host="127.0.0.1", port=port, log_level="error")
    )
    cap_task = asyncio.create_task(cap_srv.serve())

    secret_email = "real.export@enterprise.example"
    secret_phone = "415-555-0199"
    secret_key = "sk-live-9876543210abcdef1234"
    raw_input = f"status only check for {secret_email} / {secret_phone} / {secret_key}"

    old_env = {
        k: os.environ.get(k)
        for k in (
            "LANGSMITH_TRACING",
            "LANGSMITH_OTEL_ENABLED",
            "LANGSMITH_OTEL_ONLY",
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_TRACING_V2",
        )
    }

    try:
        while not cap_srv.started:
            await asyncio.sleep(0.05)

        os.environ["LANGSMITH_ENDPOINT"] = f"http://127.0.0.1:{port}"
        os.environ["LANGSMITH_API_KEY"] = "lsv2_test_local_capture_key"
        os.environ.pop("LANGCHAIN_TRACING_V2", None)

        ls_settings = Settings(telemetry_backend="langsmith", model_mode="mock")
        configure_tracer_provider(settings=ls_settings, service_name="test-ls-capture")
        callbacks = build_langchain_callbacks(ls_settings)
        assert len(callbacks) == 1

        deps = WorkflowDeps(domain_store=InMemoryDomainStore(), use_postgres=False)
        graph = build_incident_desk_graph(deps=deps, checkpointer=InMemorySaver())
        ctx = RunContext(
            tenant_id="tenant-alpha",
            requester_id="alice",
            requester_roles=frozenset({"engineer"}),
            thread_id="t-ls-capture",
            run_id="r-ls-capture",
            request_id="req-ls-capture",
        )
        await graph.ainvoke(
            {"request_text": raw_input},
            config={
                "configurable": {"thread_id": ctx.thread_id},
                "callbacks": callbacks,
            },
            context=ctx,
            durability="sync",
        )

        # Flush LangSmith background batcher in a thread so uvicorn can serve /runs/multipart
        inner = getattr(callbacks[0], "_inner", callbacks[0])
        client_obj = getattr(inner, "client", None)
        if client_obj is not None and hasattr(client_obj, "flush"):
            await asyncio.to_thread(client_obj.flush)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(
            "[REDACTED_EMAIL]" in b for b in captured_bodies
        ):
            await asyncio.sleep(0.1)

        assert len(captured_bodies) > 0, "Expected outgoing LangSmith HTTP payloads to be captured"
        all_payloads = "\n".join(captured_bodies)
        assert "[REDACTED_EMAIL]" in all_payloads, (
            "Expected masked run inputs to be exported to LangSmith capture server"
        )
        for secret in (secret_email, secret_phone, secret_key):
            assert secret not in all_payloads, f"Leaked PII ({secret}) in LangSmith HTTP export!"
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cap_srv.should_exit = True
        await cap_task
