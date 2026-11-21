"""End-to-end tests for Agent-Server-style multitask_strategy (v2.3): reject, enqueue, interrupt."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx
import psycopg

from api.main import create_app
from core.auth import mint_user_jwt
from core.config import get_settings
from core.db import apply_app_migrations
from core.models import CallerIdentity
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain


async def _reset_db() -> None:
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


def _alice_hdr() -> dict[str, str]:
    settings = get_settings()
    tok = mint_user_jwt(
        CallerIdentity(tenant_id="tenant-alpha", actor_id="alice", roles=frozenset({"engineer"})),
        secret=settings.jwt_signing_secret,
    )
    return {"Authorization": f"Bearer {tok}"}


def _spawn_worker() -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.update(
        {
            "WORKER_LEASE_S": "5",
            "WORKER_HEARTBEAT_S": "2",
            "MODEL_MODE": "mock",
            "TELEMETRY_BACKEND": "none",
        }
    )
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


async def test_multitask_strategy_reject_enqueue_and_interrupt() -> None:
    await _reset_db()
    app = create_app()
    alice = _alice_hdr()

    # 1. `enqueue`: two runs queued on the same thread before worker starts;
    #    two concurrent workers must execute them sequentially in FIFO order!
    w1: subprocess.Popen[bytes] | None = None
    w2: subprocess.Popen[bytes] | None = None
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_enq = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r1 = await client.post(
                f"/v1/threads/{t_enq}/runs",
                json={"request_text": "status only check 1", "multitask_strategy": "reject"},
                headers=alice,
            )
            assert r1.status_code == 202

            # Default `reject` returns 409
            r_rej = await client.post(
                f"/v1/threads/{t_enq}/runs",
                json={"request_text": "rejected duplicate"},
                headers=alice,
            )
            assert r_rej.status_code == 409

            # `enqueue` accepts follow-up run behind r1
            r2 = await client.post(
                f"/v1/threads/{t_enq}/runs",
                json={"request_text": "status only check 2", "multitask_strategy": "enqueue"},
                headers=alice,
            )
            assert r2.status_code == 202
            assert r2.json()["multitask_strategy"] == "enqueue"

            # 2. `interrupt`: cancels waiting queued run and enqueues replacement run
            t_int = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            ri1 = await client.post(
                f"/v1/threads/{t_int}/runs",
                json={"request_text": "status only first"},
                headers=alice,
            )
            assert ri1.status_code == 202
            ri2 = await client.post(
                f"/v1/threads/{t_int}/runs",
                json={
                    "request_text": "status only replacement",
                    "multitask_strategy": "interrupt",
                },
                headers=alice,
            )
            assert ri2.status_code == 202
            assert (
                await client.get(f"/v1/runs/{ri1.json()['run_id']}", headers=alice)
            ).json()["status"] == "cancelled"

            # Start two workers -> r1 & r2 both succeed sequentially, ri2 succeeds
            w1 = _spawn_worker()
            w2 = _spawn_worker()
            deadline = time.monotonic() + 15.0
            targets = [r1.json()["run_id"], r2.json()["run_id"], ri2.json()["run_id"]]
            statuses: dict[str, str] = {}
            while time.monotonic() < deadline:
                statuses = {
                    rid: (await client.get(f"/v1/runs/{rid}", headers=alice)).json()["status"]
                    for rid in targets
                }
                if all(st in {"succeeded", "failed", "cancelled"} for st in statuses.values()):
                    break
                await asyncio.sleep(0.2)
            assert all(st == "succeeded" for st in statuses.values()), statuses
    finally:
        _stop(w1)
        _stop(w2)
