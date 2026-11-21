"""Boundary-first regression tests for PostgreSQL connection-drop recovery.

Reproduces two real failure scenarios using `pg_terminate_backend` against live Postgres:
  1. Idle worker + pending approval: terminate all `app_rw` backends while a run is
     `awaiting_approval` -> API `decide_approval` must return `202` (zero 500s!) and the
     SAME worker process must survive, reconnect automatically, and complete the rollback once.
  2. Post-write connection drop: terminate `app_rw` backends after a write commits -> worker
     must preserve lease fencing, reconcile/recover without duplicating the mutation, and
     settle with `mock_operations` count == 1 and `checkout-api == v2.4.0`.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import psycopg

from api.main import create_app
from core.config import get_settings
from core.db import apply_app_migrations, set_tenant_context
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import PostgresDomainStore
from worker.main import RunWorker
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph


async def _terminate_app_rw_connections(admin_dsn: str) -> int:
    """Forcibly kill all active `app_rw` PostgreSQL sessions (simulates DB restart/network drop)."""
    async with (
        await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            """
            SELECT count(pg_terminate_backend(pid))
              FROM pg_stat_activity
             WHERE usename = 'app_rw'
               AND pid <> pg_backend_pid()
            """
        )
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def _poll_run(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    run_id: str,
    targets: set[str],
    *,
    timeout_s: float = 25.0,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, object] = {}
    while asyncio.get_running_loop().time() < deadline:
        r = await client.get(f"/v1/runs/{run_id}", headers=headers)
        assert r.status_code == 200, f"Unexpected HTTP {r.status_code}: {r.text}"
        last = r.json()
        if last["status"] in targets:
            return last
        await asyncio.sleep(0.15)
    raise AssertionError(f"Run {run_id} did not reach {targets}; last={last}")


async def test_worker_and_api_recover_automatically_from_dropped_postgres_connections() -> None:
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
    worker = RunWorker(settings=settings, worker_id="w-pg-drop", poll_interval_s=0.1)

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

        worker_task = asyncio.create_task(worker.serve_forever())
        try:
            # 1. Wait until Run 1 reaches awaiting_approval and worker is idle
            await _poll_run(client, alice_h, r1, {"awaiting_approval"})
            await asyncio.sleep(0.25)

            # 2. Terminate ALL app_rw Postgres connections while worker is idle & approval pending
            killed = await _terminate_app_rw_connections(settings.database_admin_url)
            assert killed >= 1

            # Give worker a beat to hit the dropped connection; it MUST stay alive (not exit!)
            await asyncio.sleep(0.3)
            assert not worker_task.done(), f"Worker crashed on DB drop: {worker_task.exception()}"

            # 3. API approval call must succeed on the VERY FIRST attempt (202, zero 500s!)
            approvals = (await client.get("/v1/approvals", headers=bob_h)).json()
            apr = next(a for a in approvals if a["thread_id"] == t_id)
            operation_id = apr["operation_id"]

            dec = await client.post(
                f"/v1/approvals/{apr['approval_id']}/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
            assert dec.status_code == 202, f"API failed after DB drop: {dec.status_code} {dec.text}"
            resume_run_id = dec.json()["run_id"]

            # 4. The SAME worker must reconnect automatically and complete the approved run
            res_state = await _poll_run(
                client, bob_h, resume_run_id, {"succeeded", "failed"}
            )
            assert res_state["status"] == "succeeded", res_state

            # 5. Verify exactly one mutation in mock_operations and final service state v2.4.0
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
                    "SELECT current_version, status FROM mock_services "
                    "WHERE service_name = 'checkout-api'"
                )
                assert await cur.fetchone() == ("v2.4.0", "healthy")
        finally:
            worker.request_stop()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker_task, timeout=5.0)


async def test_post_write_postgres_connection_drop_reclaims_and_commits_once() -> None:
    """Drop all Postgres connections right after a write commits; same worker reclaims once."""
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

    dropped_once = False

    class _DropAfterFirstWriteStore(PostgresDomainStore):
        async def execute_service_change(self, **kwargs):  # type: ignore[override]
            nonlocal dropped_once
            res = await super().execute_service_change(**kwargs)
            if not dropped_once:
                dropped_once = True
                # Commit already succeeded in Postgres; now terminate all app_rw backends!
                await _terminate_app_rw_connections(settings.database_admin_url)
            return res

    def _graph_builder(*, deps: WorkflowDeps, checkpointer, store):
        patched_deps = WorkflowDeps(
            settings=deps.settings,
            domain_store=_DropAfterFirstWriteStore(settings.database_url),
            use_postgres=True,
            use_mcp_http=False,
            attempt=deps.attempt,
        )
        return build_incident_desk_graph(
            deps=patched_deps, checkpointer=checkpointer, store=store
        )

    app = create_app(settings)
    worker = RunWorker(
        settings=settings,
        worker_id="w-pg-post-write",
        lease_s=2,
        heartbeat_s=1.0,
        poll_interval_s=0.1,
        graph_builder=_graph_builder,
    )

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

        worker_task = asyncio.create_task(worker.serve_forever())
        try:
            await _poll_run(client, alice_h, r1, {"awaiting_approval"})
            approvals = (await client.get("/v1/approvals", headers=bob_h)).json()
            apr = next(a for a in approvals if a["thread_id"] == t_id)
            operation_id = apr["operation_id"]

            dec = await client.post(
                f"/v1/approvals/{apr['approval_id']}/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
            assert dec.status_code == 202
            resume_run_id = dec.json()["run_id"]

            # The SAME worker survives the post-write DB drop (either transparently via the
            # health-checked pool on attempt 1 or via lease reclaim on attempt >= 2) and
            # settles succeeded with exactly one mutation!
            final_state = await _poll_run(
                client, bob_h, resume_run_id, {"succeeded", "failed"}, timeout_s=20.0
            )
            assert dropped_once is True
            assert not worker_task.done()
            assert final_state["status"] == "succeeded", final_state
            assert int(final_state["attempt"]) >= 1, final_state  # type: ignore[arg-type]

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
                    "SELECT current_version, status FROM mock_services "
                    "WHERE service_name = 'checkout-api'"
                )
                assert await cur.fetchone() == ("v2.4.0", "healthy")
        finally:
            worker.request_stop()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker_task, timeout=5.0)
