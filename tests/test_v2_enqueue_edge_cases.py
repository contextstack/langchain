"""Boundary-first regression tests for same-thread `enqueue` scheduling bugs.

Reproduces two real PostgreSQL concurrency/scheduling edge cases:
  1. A queued follow-up (`enqueue`) behind an incident requiring HITL approval must NOT
     preempt the pending approval or clear it from `GET /v1/approvals`; the approval
     must remain actionable, resume first once decided, and only then run the follow-up.
  2. Two workers calling `_claim_one_run` concurrently on a thread with two queued runs
     (holding Worker 1 at its commit boundary) must NOT raise `UniqueViolation` or crash
     Worker 2.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

import httpx
import psycopg
from psycopg_pool import AsyncConnectionPool

from api.main import create_app
from core.config import get_settings
from core.db import apply_app_migrations, set_system_rls_bypass
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from worker.main import RunWorker


async def _poll_run(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    run_id: str,
    targets: set[str],
    *,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        r = await client.get(f"/v1/runs/{run_id}", headers=headers)
        assert r.status_code == 200
        last = r.json()
        if last["status"] in targets:
            return last
        await asyncio.sleep(0.15)
    raise AssertionError(f"Run {run_id} did not reach {targets}; last={last}")


async def test_queued_followup_does_not_strand_pending_approval() -> None:
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

    app = create_app(settings)
    worker = RunWorker(settings=settings, worker_id="w-enqueue-hitl", poll_interval_s=0.1)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        alice_tok = (await client.post("/v1/dev/token", json={"persona": "alice"})).json()["token"]
        bob_tok = (await client.post("/v1/dev/token", json={"persona": "bob"})).json()["token"]
        alice_h = {"Authorization": f"Bearer {alice_tok}"}
        bob_h = {"Authorization": f"Bearer {bob_tok}"}

        # Create thread and enqueue Run 1 (requires approval) + Run 2 (status-only follow-up)
        t_id = (
            await client.post(
                "/v1/threads", headers=alice_h, json={"workflow_name": "incident_desk"}
            )
        ).json()["thread_id"]
        r1 = (
            await client.post(
                f"/v1/threads/{t_id}/runs",
                headers=alice_h,
                json={"request_text": "checkout-api returning 502s after v2.4.1 deploy"},
            )
        ).json()["run_id"]
        r2_resp = await client.post(
            f"/v1/threads/{t_id}/runs",
            headers=alice_h,
            json={"request_text": "status only check", "multitask_strategy": "enqueue"},
        )
        assert r2_resp.status_code == 202
        r2 = r2_resp.json()["run_id"]

        worker_task = asyncio.create_task(worker.serve_forever())
        try:
            # Run 1 must pause at awaiting_approval
            await _poll_run(client, alice_h, r1, {"awaiting_approval"})
            # Give the worker several poll cycles to make sure it does NOT prematurely run r2
            await asyncio.sleep(0.8)

            # Thread must still be awaiting_approval, r2 still queued, approval visible in inbox
            threads = (await client.get("/v1/threads", headers=alice_h)).json()
            t_state = next(t for t in threads if t["thread_id"] == t_id)
            assert t_state["status"] == "awaiting_approval", t_state

            r2_state = (await client.get(f"/v1/runs/{r2}", headers=alice_h)).json()
            assert r2_state["status"] == "queued", r2_state

            inbox = (await client.get("/v1/approvals", headers=bob_h)).json()
            matching = [a for a in inbox if a["thread_id"] == t_id]
            assert len(matching) == 1, f"Approval stranded from inbox: {inbox}"
            approval_id = matching[0]["approval_id"]

            # Approve Run 1 -> resume run must execute BEFORE queued follow-up r2
            dec = await client.post(
                f"/v1/approvals/{approval_id}/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
            assert dec.status_code == 202
            resume_run_id = dec.json()["run_id"]

            res_state = await _poll_run(client, bob_h, resume_run_id, {"succeeded", "failed"})
            assert res_state["status"] == "succeeded", res_state

            # Now the queued follow-up r2 should run and succeed
            r2_final = await _poll_run(client, alice_h, r2, {"succeeded", "failed"})
            assert r2_final["status"] == "succeeded", r2_final
        finally:
            worker.request_stop()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker_task, timeout=5.0)


async def test_cancel_queued_followup_preserves_pending_approval() -> None:
    """Cancelling a queued follow-up behind a pending approval keeps thread awaiting_approval."""
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

    app = create_app(settings)
    worker = RunWorker(settings=settings, worker_id="w-cancel-followup", poll_interval_s=0.1)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        alice_tok = (await client.post("/v1/dev/token", json={"persona": "alice"})).json()["token"]
        bob_tok = (await client.post("/v1/dev/token", json={"persona": "bob"})).json()["token"]
        alice_h = {"Authorization": f"Bearer {alice_tok}"}
        bob_h = {"Authorization": f"Bearer {bob_tok}"}

        t_id = (
            await client.post(
                "/v1/threads", headers=alice_h, json={"workflow_name": "incident_desk"}
            )
        ).json()["thread_id"]
        r1 = (
            await client.post(
                f"/v1/threads/{t_id}/runs",
                headers=alice_h,
                json={"request_text": "checkout-api returning 502s after v2.4.1 deploy"},
            )
        ).json()["run_id"]
        r2 = (
            await client.post(
                f"/v1/threads/{t_id}/runs",
                headers=alice_h,
                json={"request_text": "status only check", "multitask_strategy": "enqueue"},
            )
        ).json()["run_id"]

        worker_task = asyncio.create_task(worker.serve_forever())
        try:
            await _poll_run(client, alice_h, r1, {"awaiting_approval"})

            # Cancel the queued follow-up r2 while r1's approval is pending
            cancel_resp = await client.post(f"/v1/runs/{r2}/cancel", headers=alice_h)
            assert cancel_resp.status_code == 200
            cancel_body = cancel_resp.json()
            assert cancel_body["status"] == "cancelled"
            assert cancel_body["thread_status"] == "awaiting_approval", cancel_body

            # Thread must still be awaiting_approval and approval must remain in inbox
            threads = (await client.get("/v1/threads", headers=alice_h)).json()
            t_state = next(t for t in threads if t["thread_id"] == t_id)
            assert t_state["status"] == "awaiting_approval", t_state

            inbox = (await client.get("/v1/approvals", headers=bob_h)).json()
            matching = [a for a in inbox if a["thread_id"] == t_id]
            assert len(matching) == 1, f"Approval hidden after cancelling follow-up: {inbox}"

            # Approver decision must succeed (202, not 409) and complete the rollback
            dec = await client.post(
                f"/v1/approvals/{matching[0]['approval_id']}/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
            assert dec.status_code == 202, dec.text
            res_state = await _poll_run(
                client, bob_h, dec.json()["run_id"], {"succeeded", "failed"}
            )
            assert res_state["status"] == "succeeded", res_state
        finally:
            worker.request_stop()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker_task, timeout=5.0)


async def test_concurrent_claim_on_same_thread_does_not_raise_unique_violation() -> None:
    """Reproduce two workers claiming queued runs on the same thread at the commit boundary."""
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )

    t1 = f"thr_race_{uuid.uuid4().hex[:8]}"
    r1 = f"run_race_1_{uuid.uuid4().hex[:8]}"
    r2 = f"run_race_2_{uuid.uuid4().hex[:8]}"

    async with (
        await psycopg.AsyncConnection.connect(settings.database_url) as conn,
        conn.transaction(),
    ):
        await set_system_rls_bypass(conn)
        await conn.execute(
            """
            INSERT INTO app_threads
                (thread_id, tenant_id, created_by, creator_roles,
                 workflow_name, release_version, checkpoint_compat, status)
            VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                    'incident_desk', '0.1.0+test', 1, 'running')
            """,
            (t1,),
        )
        await conn.execute(
            """
            INSERT INTO runs (run_id, thread_id, tenant_id, kind, status, input)
            VALUES
                (%s, %s, 'tenant-alpha', 'start', 'queued', '{"request_text":"first"}'::jsonb),
                (%s, %s, 'tenant-alpha', 'start', 'queued', '{"request_text":"second"}'::jsonb)
            """,
            (r1, t1, r2, t1),
        )

    w1_reached_commit = asyncio.Event()
    allow_w1_commit = asyncio.Event()

    class _PausingConnection:
        """Proxies an AsyncConnection and pauses right before commit()."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        async def commit(self) -> None:
            w1_reached_commit.set()
            await allow_w1_commit.wait()
            await self._inner.commit()

    class _PausingPool:
        def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
            self._pool = pool

        @contextlib.asynccontextmanager
        async def connection(self):
            async with self._pool.connection() as conn:
                yield _PausingConnection(conn)

    pool1: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=settings.database_url, min_size=1, max_size=2, open=False
    )
    pool2: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=settings.database_url, min_size=1, max_size=2, open=False
    )
    await pool1.open()
    await pool2.open()
    try:
        w1 = RunWorker(settings=settings, worker_id="worker-race-1")
        w2 = RunWorker(settings=settings, worker_id="worker-race-2")

        # Worker 1 claims Run 1 and pauses right before COMMIT
        task1 = asyncio.create_task(w1._claim_one_run(_PausingPool(pool1)))  # type: ignore[arg-type]
        await asyncio.wait_for(w1_reached_commit.wait(), timeout=5.0)

        # While Worker 1 holds the uncommitted claim, Worker 2 tries to claim
        task2 = asyncio.create_task(w2._claim_one_run(pool2))
        await asyncio.sleep(0.2)

        # Release Worker 1's commit; Worker 2 must NOT raise UniqueViolation
        allow_w1_commit.set()
        claimed1, claimed2 = await asyncio.gather(task1, task2)

        assert claimed1 is not None
        assert claimed1.run_id == r1
        assert claimed2 is None
    finally:
        await pool1.close()
        await pool2.close()
