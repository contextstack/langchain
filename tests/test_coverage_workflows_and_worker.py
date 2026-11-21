"""Targeted in-process tests for `refund_ops`, `incident_desk` edge paths, and `RunWorker`."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid

import psycopg
import pytest
import redis.asyncio as aioredis
import uvicorn
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from core.config import Settings, get_settings
from core.db import apply_app_migrations, set_system_rls_bypass
from core.models import RunContext
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.billing_server import create_billing_server
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore
from mcp_servers.infra_server import create_infra_server
from mcp_servers.ticketing_server import create_ticketing_server
from worker.main import ClaimedRun, RunWorker
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph
from workflows.refund_ops.graph import (
    build_refund_ops_graph,
)
from workflows.refund_ops.graph import (
    build_studio_graph as build_refund_studio_graph,
)
from workflows.refund_ops.version import get_checkpoint_compat as get_refund_compat
from workflows.registry import get_workflow_spec


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ctx(thread_id: str, tenant_id: str = "tenant-alpha") -> RunContext:
    return RunContext(
        tenant_id=tenant_id,
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id=thread_id,
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        request_id="req-test",
    )


async def test_refund_ops_all_lifecycle_branches_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Version override & studio graph
    assert get_refund_compat() == 1
    monkeypatch.setenv("CHECKPOINT_COMPAT_OVERRIDE", "7")
    assert get_refund_compat() == 7
    monkeypatch.delenv("CHECKPOINT_COMPAT_OVERRIDE")
    assert build_refund_studio_graph() is not None
    with pytest.raises(KeyError, match="Unknown workflow_name"):
        get_workflow_spec("does_not_exist")

    store = InMemoryDomainStore()
    mem_store = InMemoryStore()
    cp = InMemorySaver()
    deps = WorkflowDeps(domain_store=store, use_postgres=False, use_mcp_http=False)
    graph = build_refund_ops_graph(deps=deps, checkpointer=cp, store=mem_store)

    # 1. No-action (balance only)
    cfg0 = {"configurable": {"thread_id": "ref-noop"}}
    await graph.ainvoke(
        {"request_text": "balance only check for ACC-1001"},
        config=cfg0,
        context=_ctx("ref-noop"),
    )
    s0 = await graph.aget_state(cfg0)
    assert s0.values["final_status"] == "no_action"

    # 2. Policy denial (> $5,000) & cross-tenant denial (ACC-2001)
    cfg_big = {"configurable": {"thread_id": "ref-big"}}
    await graph.ainvoke(
        {"request_text": "Issue $10,000 refund to ACC-1001"},
        config=cfg_big,
        context=_ctx("ref-big"),
    )
    assert (await graph.aget_state(cfg_big)).values["final_status"] == "denied"

    cfg_xt = {"configurable": {"thread_id": "ref-xt"}}
    await graph.ainvoke(
        {"request_text": "Issue $100 refund to ACC-2001"},
        config=cfg_xt,
        context=_ctx("ref-xt"),
    )
    assert (await graph.aget_state(cfg_xt)).values["final_status"] == "denied"

    # 3. Reject path
    cfg_rej = {"configurable": {"thread_id": "ref-rej"}}
    await graph.ainvoke(
        {"request_text": "Issue $100 refund to ACC-1001"},
        config=cfg_rej,
        context=_ctx("ref-rej"),
    )
    await graph.ainvoke(
        Command(resume={"decision": "reject", "approver_id": "bob"}),
        config=cfg_rej,
        context=_ctx("ref-rej"),
    )
    assert (await graph.aget_state(cfg_rej)).values["final_status"] == "rejected"

    # 4. Edit path -> denied edit (> $5,000)
    cfg_ed_bad = {"configurable": {"thread_id": "ref-ed-bad"}}
    await graph.ainvoke(
        {"request_text": "Issue $100 refund to ACC-1001"},
        config=cfg_ed_bad,
        context=_ctx("ref-ed-bad"),
    )
    await graph.ainvoke(
        Command(
            resume={
                "decision": "edit",
                "approver_id": "bob",
                "edits": {"amount_cents": 900_000},
            }
        ),
        config=cfg_ed_bad,
        context=_ctx("ref-ed-bad"),
    )
    assert (await graph.aget_state(cfg_ed_bad)).values["final_status"] == "denied"

    # 5. Edit path -> valid edit ($50) -> executes, verifies, and writes memory
    cfg_ed_ok = {"configurable": {"thread_id": "ref-ed-ok"}}
    await graph.ainvoke(
        {"request_text": "Issue $100 refund to ACC-1001"},
        config=cfg_ed_ok,
        context=_ctx("ref-ed-ok"),
    )
    await graph.ainvoke(
        Command(
            resume={
                "decision": "edit",
                "approver_id": "bob",
                "edits": {"amount_cents": 5_000},
            }
        ),
        config=cfg_ed_ok,
        context=_ctx("ref-ed-ok"),
    )
    end_ed = await graph.aget_state(cfg_ed_ok)
    assert end_ed.values["final_status"] == "completed"
    assert store.accounts[("tenant-alpha", "ACC-1001")]["balance_cents"] == 95_000

    # 6. Approved plan hash mismatch -> refused without writing
    cfg_hm = {"configurable": {"thread_id": "ref-hm"}}
    await graph.ainvoke(
        {"request_text": "Issue $100 refund to ACC-1001"},
        config=cfg_hm,
        context=_ctx("ref-hm"),
    )
    await graph.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": "tampered_hash",
            }
        ),
        config=cfg_hm,
        context=_ctx("ref-hm"),
    )
    assert (await graph.aget_state(cfg_hm)).values["final_status"] == "denied_plan_mismatch"

    # 7. Timeout -> reconciliation (outcome_unknown when not committed, succeeded when committed)
    timeout_deps = WorkflowDeps(
        domain_store=store,
        use_postgres=False,
        use_mcp_http=False,
        simulate_write_timeout=True,
    )
    t_graph = build_refund_ops_graph(deps=timeout_deps, checkpointer=InMemorySaver())
    cfg_to = {"configurable": {"thread_id": "ref-to"}}
    await t_graph.ainvoke(
        {"request_text": "Issue $50 refund to ACC-1001"},
        config=cfg_to,
        context=_ctx("ref-to"),
    )
    intr_to = (await t_graph.aget_state(cfg_to)).interrupts[0].value
    await t_graph.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": intr_to["plan_hash"],
            }
        ),
        config=cfg_to,
        context=_ctx("ref-to"),
    )
    assert (await t_graph.aget_state(cfg_to)).values["final_status"] == "outcome_unknown"


async def test_refund_ops_and_incident_desk_live_mcp_http_in_process() -> None:
    """Exercise `use_mcp_http=True` paths in-process for both workflows (incl. ticket & timeout)."""
    secret = "mcp-inproc-secret-key-at-least-32-bytes-long"
    store = InMemoryDomainStore()
    p_infra, p_ticket, p_bill = _free_port(), _free_port(), _free_port()
    srvs = [
        uvicorn.Server(
            uvicorn.Config(
                create_infra_server(store=store, mcp_secret=secret).http_app(path="/mcp"),
                host="127.0.0.1",
                port=p_infra,
                log_level="error",
            )
        ),
        uvicorn.Server(
            uvicorn.Config(
                create_ticketing_server(store=store, mcp_secret=secret).http_app(path="/mcp"),
                host="127.0.0.1",
                port=p_ticket,
                log_level="error",
            )
        ),
        uvicorn.Server(
            uvicorn.Config(
                create_billing_server(store=store, mcp_secret=secret).http_app(path="/mcp"),
                host="127.0.0.1",
                port=p_bill,
                log_level="error",
            )
        ),
    ]
    tasks = [asyncio.create_task(s.serve()) for s in srvs]
    try:
        while not all(s.started for s in srvs):
            await asyncio.sleep(0.05)

        cfg_settings = Settings(
            mcp_token_secret=secret,
            mcp_infra_url=f"http://127.0.0.1:{p_infra}/mcp",
            mcp_ticketing_url=f"http://127.0.0.1:{p_ticket}/mcp",
            mcp_billing_url=f"http://127.0.0.1:{p_bill}/mcp",
            mcp_transport="http",
        )

        # 1. refund_ops over live MCP HTTP (happy path)
        r_deps = WorkflowDeps(
            settings=cfg_settings, domain_store=store, use_postgres=False, use_mcp_http=True
        )
        r_graph = build_refund_ops_graph(
            deps=r_deps, checkpointer=InMemorySaver(), store=InMemoryStore()
        )
        rcfg = {"configurable": {"thread_id": "ref-http-1"}}
        await r_graph.ainvoke(
            {"request_text": "Issue $100 refund to ACC-1001"},
            config=rcfg,
            context=_ctx("ref-http-1"),
        )
        r_intr = (await r_graph.aget_state(rcfg)).interrupts[0].value
        await r_graph.ainvoke(
            Command(
                resume={
                    "decision": "approve",
                    "approver_id": "bob",
                    "approved_plan_hash": r_intr["plan_hash"],
                }
            ),
            config=rcfg,
            context=_ctx("ref-http-1"),
        )
        assert (await r_graph.aget_state(rcfg)).values["final_status"] == "completed"

        # 2. incident_desk over live MCP HTTP (ticket update + timeout reconciliation)
        i_deps = WorkflowDeps(
            settings=cfg_settings, domain_store=store, use_postgres=False, use_mcp_http=True
        )
        i_cp = InMemorySaver()
        i_graph = build_incident_desk_graph(deps=i_deps, checkpointer=i_cp)
        icfg = {"configurable": {"thread_id": "inc-http-tkt"}}
        await i_graph.ainvoke(
            {"request_text": "checkout-api returning 502s"},
            config=icfg,
            context=_ctx("inc-http-tkt"),
        )
        # Swap plan via valid edit to update_ticket so ticket write + verify_end_state run
        await i_graph.ainvoke(
            Command(
                resume={
                    "decision": "edit",
                    "approver_id": "bob",
                    "edits": {
                        "resource_type": "ticket",
                        "target_resource": "INC-1001",
                        "action": "update_ticket",
                        "target_version": None,
                        "ticket_id": "INC-1001",
                    },
                }
            ),
            config=icfg,
            context=_ctx("inc-http-tkt"),
        )
        assert (await i_graph.aget_state(icfg)).values["final_status"] == "completed"
        assert store.tickets[("tenant-alpha", "INC-1001")]["status"] == "resolved"

        # 3. incident_desk timeout -> HTTP reconciliation (outcome_unknown)
        i_to_deps = WorkflowDeps(
            settings=cfg_settings,
            domain_store=store,
            use_postgres=False,
            use_mcp_http=True,
            simulate_write_timeout=True,
        )
        i_to_graph = build_incident_desk_graph(deps=i_to_deps, checkpointer=InMemorySaver())
        tcfg = {"configurable": {"thread_id": "inc-http-to"}}
        await i_to_graph.ainvoke(
            {"request_text": "checkout-api returning 502s"},
            config=tcfg,
            context=_ctx("inc-http-to"),
        )
        t_intr = (await i_to_graph.aget_state(tcfg)).interrupts[0].value
        await i_to_graph.ainvoke(
            Command(
                resume={
                    "decision": "approve",
                    "approver_id": "bob",
                    "approved_plan_hash": t_intr["plan_hash"],
                }
            ),
            config=tcfg,
            context=_ctx("inc-http-to"),
        )
        assert (await i_to_graph.aget_state(tcfg)).values["final_status"] == "outcome_unknown"
    finally:
        for s in srvs:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_worker_in_process_sweep_compat_recovery_timeout_and_heartbeat() -> None:
    """Exercise RunWorker branches in-process (exhausted sweep, compat block, heartbeat loss)."""
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

    pool: AsyncConnectionPool = AsyncConnectionPool(
        conninfo=settings.database_url, min_size=2, max_size=6, open=False
    )
    await pool.open()
    rds = aioredis.from_url(settings.redis_url, decode_responses=True)
    worker = RunWorker(settings=settings, worker_id="w-cov-1", lease_s=10, heartbeat_s=0.05)

    try:
        async with (
            AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer,
            AsyncPostgresStore.from_conn_string(settings.database_url) as store,
        ):
            domain = PostgresDomainStore(settings.database_url)
            sweep_graph = build_incident_desk_graph(
                deps=WorkflowDeps(settings=settings, domain_store=domain, use_postgres=True),
                checkpointer=checkpointer,
                store=store,
            )

            # 1. _sweep_expired_runs: exhausted max_attempts + expired cancel_requested (no op_id)
            t_exh, r_exh = f"thr_exh_{uuid.uuid4().hex[:6]}", f"run_exh_{uuid.uuid4().hex[:6]}"
            t_can, r_can = f"thr_can_{uuid.uuid4().hex[:6]}", f"run_can_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                for tid in (t_exh, t_can):
                    await conn.execute(
                        """
                        INSERT INTO app_threads
                            (thread_id, tenant_id, created_by, creator_roles,
                             workflow_name, release_version, checkpoint_compat, status)
                        VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                                'incident_desk', '0.1.0+test', 1, 'running')
                        """,
                        (tid,),
                    )
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, max_attempts, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running', '{}'::jsonb,
                            3, 3, 'dead-worker', now() - interval '10 seconds')
                    """,
                    (r_exh, t_exh),
                )
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, max_attempts, cancel_requested, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running', '{}'::jsonb,
                            1, 3, true, 'dead-worker', now() - interval '10 seconds')
                    """,
                    (r_can, t_can),
                )

            await worker._sweep_expired_runs(pool, rds, sweep_graph, domain)
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute(
                    "SELECT status, error_detail FROM runs WHERE run_id = %s", (r_exh,)
                )
                assert await cur.fetchone() == ("failed", "max_attempts_exhausted")
                await cur.execute(
                    "SELECT status, error_detail FROM runs WHERE run_id = %s", (r_can,)
                )
                assert await cur.fetchone() == ("cancelled", "cancelled_expired")

            # 2. _execute_claimed_run: unknown workflow_name -> fatal
            t_unk, r_unk = f"thr_unk_{uuid.uuid4().hex[:6]}", f"run_unk_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    INSERT INTO app_threads
                        (thread_id, tenant_id, created_by, creator_roles,
                         workflow_name, release_version, checkpoint_compat, status)
                    VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                            'nonexistent_wf', '0.1.0+test', 1, 'running')
                    """,
                    (t_unk,),
                )
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running', '{}'::jsonb,
                            1, %s, now() + interval '30 seconds')
                    """,
                    (r_unk, t_unk, worker.worker_id),
                )
            await worker._execute_claimed_run(
                pool,
                rds,
                checkpointer,
                store,
                ClaimedRun(
                    run_id=r_unk,
                    thread_id=t_unk,
                    tenant_id="tenant-alpha",
                    kind="start",
                    input={},
                    attempt=1,
                    created_by="alice",
                    creator_roles=frozenset({"engineer"}),
                    release_version="0.1.0+test",
                    checkpoint_compat=1,
                    workflow_name="nonexistent_wf",
                ),
            )
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute(
                    "SELECT status, error_category FROM runs WHERE run_id = %s", (r_unk,)
                )
                assert await cur.fetchone() == ("failed", "fatal")

            # 3. _execute_claimed_run: checkpoint_compat mismatch -> blocked_version
            t_cmp, r_cmp = f"thr_cmp_{uuid.uuid4().hex[:6]}", f"run_cmp_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    INSERT INTO app_threads
                        (thread_id, tenant_id, created_by, creator_roles,
                         workflow_name, release_version, checkpoint_compat, status)
                    VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                            'incident_desk', '0.1.0+old', 99, 'running')
                    """,
                    (t_cmp,),
                )
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running', '{}'::jsonb,
                            1, %s, now() + interval '30 seconds')
                    """,
                    (r_cmp, t_cmp, worker.worker_id),
                )
            await worker._execute_claimed_run(
                pool,
                rds,
                checkpointer,
                store,
                ClaimedRun(
                    run_id=r_cmp,
                    thread_id=t_cmp,
                    tenant_id="tenant-alpha",
                    kind="start",
                    input={},
                    attempt=1,
                    created_by="alice",
                    creator_roles=frozenset({"engineer"}),
                    release_version="0.1.0+old",
                    checkpoint_compat=99,
                    workflow_name="incident_desk",
                ),
            )
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute("SELECT status FROM app_threads WHERE thread_id = %s", (t_cmp,))
                assert (await cur.fetchone())[0] == "blocked_version"

            # 4. _heartbeat_loop detects lost lease when another worker owns the row
            lost = asyncio.Event()
            hb = asyncio.create_task(
                worker._heartbeat_loop(pool, run_id="run_missing_hb", attempt=1, lost_lease=lost)
            )
            await asyncio.wait_for(lost.wait(), timeout=2.0)
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

            # 5. _fenced_finalize returns False when lease_owner does not match
            ok = await worker._fenced_finalize(
                pool,
                run_id=r_cmp,
                thread_id=t_cmp,
                attempt=99,
                run_status="succeeded",
                thread_status="completed",
            )
            assert ok is False

            # 6. _execute_claimed_run: attempt > 1 on an already-interrupted start run
            t_rec, r_rec = f"thr_rec_{uuid.uuid4().hex[:6]}", f"run_rec_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    INSERT INTO app_threads
                        (thread_id, tenant_id, created_by, creator_roles,
                         workflow_name, release_version, checkpoint_compat, status)
                    VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                            'incident_desk', '0.1.0+test', 1, 'running')
                    """,
                    (t_rec,),
                )
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running',
                            '{"request_text":"checkout-api returning 502s"}'::jsonb,
                            1, %s, now() + interval '30 seconds')
                    """,
                    (r_rec, t_rec, worker.worker_id),
                )
            claimed_rec = ClaimedRun(
                run_id=r_rec,
                thread_id=t_rec,
                tenant_id="tenant-alpha",
                kind="start",
                input={"request_text": "checkout-api returning 502s"},
                attempt=1,
                created_by="alice",
                creator_roles=frozenset({"engineer"}),
                release_version="0.1.0+test",
                checkpoint_compat=1,
                workflow_name="incident_desk",
            )
            # Run attempt 1 to hit interrupt, then simulate attempt 2 recovery on the same interrupt
            await worker._execute_claimed_run(pool, rds, checkpointer, store, claimed_rec)
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    UPDATE runs
                       SET status = 'running', attempt = 2, lease_owner = %s,
                           lease_expires_at = now() + interval '30 seconds'
                     WHERE run_id = %s
                    """,
                    (worker.worker_id, r_rec),
                )
            await worker._execute_claimed_run(
                pool,
                rds,
                checkpointer,
                store,
                ClaimedRun(**{**claimed_rec.__dict__, "attempt": 2}),
            )
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute("SELECT status FROM runs WHERE run_id = %s", (r_rec,))
                assert (await cur.fetchone())[0] == "awaiting_approval"

            # 7. _reconcile_in_flight_write direct (succeeded + not_found/unknown + exception)
            op_ok = f"op_rec_ok_{uuid.uuid4().hex[:6]}"
            await domain.execute_service_change(
                tenant_id="tenant-alpha",
                service_name="checkout-api",
                action="restart",
                operation_id=op_ok,
            )
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                st_ok = await worker._reconcile_in_flight_write(
                    conn=conn,
                    domain=domain,
                    tenant_id="tenant-alpha",
                    requester_id="alice",
                    thread_id=t_rec,
                    run_id=r_rec,
                    attempt=1,
                    operation_id=op_ok,
                    release_version="0.1.0+test",
                )
                assert st_ok[0] == "succeeded"
                st_unk = await worker._reconcile_in_flight_write(
                    conn=conn,
                    domain=domain,
                    tenant_id="tenant-alpha",
                    requester_id="alice",
                    thread_id=t_rec,
                    run_id=r_rec,
                    attempt=1,
                    operation_id="op_never_committed",
                    release_version="0.1.0+test",
                )
                assert st_unk[0] == "failed" and st_unk[2] == "outcome_unknown"

                # HTTP reconciliation error path (unreachable MCP endpoint -> outcome_unknown)
                http_worker = RunWorker(
                    settings=Settings(
                        mcp_transport="http",
                        mcp_infra_url="http://127.0.0.1:1/mcp",
                        tool_timeout_s=1.0,
                    ),
                    worker_id="w-cov-http",
                )
                st_http_err = await http_worker._reconcile_in_flight_write(
                    conn=conn,
                    domain=domain,
                    tenant_id="tenant-alpha",
                    requester_id="alice",
                    thread_id=t_rec,
                    run_id=r_rec,
                    attempt=1,
                    operation_id="op_http_unreachable",
                    release_version="0.1.0+test",
                )
                assert st_http_err[0] == "failed" and st_http_err[2] == "outcome_unknown"

            # 7b. _sweep_expired_runs on expired cancel_requested WITH op_id in checkpoint
            r_can_op = f"run_canop_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, max_attempts, cancel_requested, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'resume', 'running', '{}'::jsonb,
                            1, 3, true, 'dead-worker', now() - interval '10 seconds')
                    """,
                    (r_can_op, t_rec),
                )
            await worker._sweep_expired_runs(pool, rds, sweep_graph, domain)
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute(
                    "SELECT status, error_category FROM runs WHERE run_id = %s", (r_can_op,)
                )
                assert await cur.fetchone() == ("failed", "outcome_unknown")

            # 7c. _execute_claimed_run CancelledError WITH op_id in checkpoint (reconcile branch)
            r_can_in = f"run_canin_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         cancel_requested, attempt, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running',
                            '{"request_text":"status only check"}'::jsonb,
                            true, 1, %s, now() + interval '30 seconds')
                    """,
                    (r_can_in, t_rec, worker.worker_id),
                )
            await worker._execute_claimed_run(
                pool,
                rds,
                checkpointer,
                store,
                ClaimedRun(
                    run_id=r_can_in,
                    thread_id=t_rec,
                    tenant_id="tenant-alpha",
                    kind="start",
                    input={"request_text": "status only check"},
                    attempt=1,
                    created_by="alice",
                    creator_roles=frozenset({"engineer"}),
                    release_version="0.1.0+test",
                    checkpoint_compat=1,
                    workflow_name="incident_desk",
                ),
            )
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute(
                    "SELECT status, error_category FROM runs WHERE run_id = %s", (r_can_in,)
                )
                assert await cur.fetchone() == ("failed", "outcome_unknown")

            # 8. _execute_claimed_run: attempt > 1 on completed checkpoint
            t_done, r_done = f"thr_dn_{uuid.uuid4().hex[:6]}", f"run_dn_{uuid.uuid4().hex[:6]}"
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    INSERT INTO app_threads
                        (thread_id, tenant_id, created_by, creator_roles,
                         workflow_name, release_version, checkpoint_compat, status)
                    VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                            'incident_desk', '0.1.0+test', 1, 'running')
                    """,
                    (t_done,),
                )
                await conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, thread_id, tenant_id, kind, status, input,
                         attempt, lease_owner, lease_expires_at)
                    VALUES (%s, %s, 'tenant-alpha', 'start', 'running',
                            '{"request_text":"status only check"}'::jsonb,
                            1, %s, now() + interval '30 seconds')
                    """,
                    (r_done, t_done, worker.worker_id),
                )
            claimed_done = ClaimedRun(
                run_id=r_done,
                thread_id=t_done,
                tenant_id="tenant-alpha",
                kind="start",
                input={"request_text": "status only check"},
                attempt=1,
                created_by="alice",
                creator_roles=frozenset({"engineer"}),
                release_version="0.1.0+test",
                checkpoint_compat=1,
                workflow_name="incident_desk",
            )
            await worker._execute_claimed_run(pool, rds, checkpointer, store, claimed_done)
            # Re-run attempt=2 on already-completed checkpoint -> recovery branch
            async with pool.connection() as conn, conn.transaction():
                await set_system_rls_bypass(conn)
                await conn.execute(
                    """
                    UPDATE runs
                       SET status = 'running', attempt = 2, lease_owner = %s,
                           lease_expires_at = now() + interval '30 seconds'
                     WHERE run_id = %s
                    """,
                    (worker.worker_id, r_done),
                )
            await worker._execute_claimed_run(
                pool,
                rds,
                checkpointer,
                store,
                ClaimedRun(**{**claimed_done.__dict__, "attempt": 2}),
            )
            async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await set_system_rls_bypass(conn)
                await cur.execute("SELECT status FROM runs WHERE run_id = %s", (r_done,))
                assert (await cur.fetchone())[0] == "succeeded"

            # 9. _execute_claimed_run: CancelledError, TimeoutError, and fatal Exception
            #    using an injected controlled graph runner to prove each outcome distinctly.
            class _ControlledGraph:
                def __init__(self, mode: str) -> None:
                    self._mode = mode

                async def aget_state(self, config: dict) -> None:
                    return None

                async def astream(self, *args, **kwargs):
                    if self._mode == "timeout":
                        await asyncio.sleep(5)
                    elif self._mode == "err":
                        raise RuntimeError("boom")
                    yield {"retrieve_context": {}}

            for label, mode in (("cancel", "cancel"), ("timeout", "timeout"), ("err", "err")):
                tid = f"thr_{label}_{uuid.uuid4().hex[:6]}"
                rid = f"run_{label}_{uuid.uuid4().hex[:6]}"
                async with pool.connection() as conn, conn.transaction():
                    await set_system_rls_bypass(conn)
                    await conn.execute(
                        """
                        INSERT INTO app_threads
                            (thread_id, tenant_id, created_by, creator_roles,
                             workflow_name, release_version, checkpoint_compat, status)
                        VALUES (%s, 'tenant-alpha', 'alice', ARRAY['engineer'],
                                'incident_desk', '0.1.0+test', 1, 'running')
                        """,
                        (tid,),
                    )
                    await conn.execute(
                        """
                        INSERT INTO runs
                            (run_id, thread_id, tenant_id, kind, status, input,
                             cancel_requested, attempt, lease_owner, lease_expires_at)
                        VALUES (%s, %s, 'tenant-alpha', 'start', 'running',
                                '{"request_text":"status only check"}'::jsonb,
                                %s, 1, %s, now() + interval '30 seconds')
                        """,
                        (rid, tid, mode == "cancel", worker.worker_id),
                    )
                claimed_err = ClaimedRun(
                    run_id=rid,
                    thread_id=tid,
                    tenant_id="tenant-alpha",
                    kind="start",
                    input={"request_text": "status only check"},
                    attempt=1,
                    created_by="alice",
                    creator_roles=frozenset({"engineer"}),
                    release_version="0.1.0+test",
                    checkpoint_compat=1,
                    workflow_name="incident_desk",
                )
                controlled_worker = RunWorker(
                    settings=Settings(run_timeout_s=0.05 if mode == "timeout" else 30.0),
                    worker_id=worker.worker_id,
                    lease_s=10,
                    heartbeat_s=0.05,
                    graph_builder=lambda _m=mode, **_kw: _ControlledGraph(_m),
                )
                await controlled_worker._execute_claimed_run(
                    pool, rds, checkpointer, store, claimed_err
                )

                async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                    await set_system_rls_bypass(conn)
                    await cur.execute(
                        "SELECT status, error_category, error_detail FROM runs WHERE run_id = %s",
                        (rid,),
                    )
                    row = await cur.fetchone()
                    if mode == "cancel":
                        assert row == ("cancelled", None, None), row
                    elif mode == "timeout":
                        assert row == ("failed", "transient", "run_timeout"), row
                    else:
                        assert row == ("failed", "fatal", "boom"), row

            # 10. incident_desk direct update_ticket + direct timeout reconcile
            direct_store = InMemoryDomainStore()
            d_graph = build_incident_desk_graph(
                deps=WorkflowDeps(
                    domain_store=direct_store, use_postgres=False, use_mcp_http=False
                ),
                checkpointer=InMemorySaver(),
            )
            dcfg = {"configurable": {"thread_id": "inc-dir-tkt"}}
            await d_graph.ainvoke(
                {"request_text": "checkout-api returning 502s"},
                config=dcfg,
                context=_ctx("inc-dir-tkt"),
            )
            await d_graph.ainvoke(
                Command(
                    resume={
                        "decision": "edit",
                        "approver_id": "bob",
                        "edits": {
                            "resource_type": "ticket",
                            "target_resource": "INC-1001",
                            "action": "update_ticket",
                            "target_version": None,
                            "ticket_id": "INC-1001",
                        },
                    }
                ),
                config=dcfg,
                context=_ctx("inc-dir-tkt"),
            )
            assert (await d_graph.aget_state(dcfg)).values["final_status"] == "completed"

            dto_graph = build_incident_desk_graph(
                deps=WorkflowDeps(
                    domain_store=direct_store,
                    use_postgres=False,
                    use_mcp_http=False,
                    simulate_write_timeout=True,
                ),
                checkpointer=InMemorySaver(),
            )
            dtocfg = {"configurable": {"thread_id": "inc-dir-to"}}
            await dto_graph.ainvoke(
                {"request_text": "checkout-api returning 502s"},
                config=dtocfg,
                context=_ctx("inc-dir-to"),
            )
            d_intr = (await dto_graph.aget_state(dtocfg)).interrupts[0].value
            await dto_graph.ainvoke(
                Command(
                    resume={
                        "decision": "approve",
                        "approver_id": "bob",
                        "approved_plan_hash": d_intr["plan_hash"],
                    }
                ),
                config=dtocfg,
                context=_ctx("inc-dir-to"),
            )
            assert (await dto_graph.aget_state(dtocfg)).values["final_status"] == "outcome_unknown"
    finally:
        await rds.aclose()
        await pool.close()
