"""v2.6 boundary-first test: cross-thread tenant-scoped memory via AsyncPostgresStore.

Verifies over real PostgreSQL (`AsyncPostgresStore` + `AsyncPostgresSaver`):
  1. Thread 1 (`tenant-alpha`) completes a verified rollback and persists its resolution
     summary into `AsyncPostgresStore` under `("tenant-alpha", "memory", "incident_desk")`.
  2. Thread 2 (`tenant-alpha`, brand-new `thread_id`) recalls Thread 1's resolution in
     `state["prior_memories"]` during `retrieve_context`.
  3. Thread 3 (`tenant-beta`, brand-new `thread_id`) sees 0 memories from `tenant-alpha`,
     enforcing strict cross-tenant isolation in the LangGraph Store.
"""

from __future__ import annotations

import uuid

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command

from core.config import get_settings
from core.db import apply_app_migrations, set_tenant_context
from core.models import RunContext
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph


async def test_cross_thread_memory_persists_and_isolates_tenants_in_postgres_store() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute("TRUNCATE TABLE store RESTART IDENTITY CASCADE;")
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    async with (
        AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer,
        AsyncPostgresStore.from_conn_string(settings.database_url) as store,
    ):
        graph = build_incident_desk_graph(
            deps=WorkflowDeps(settings=settings, use_postgres=True, use_mcp_http=False),
            checkpointer=checkpointer,
            store=store,
        )

        # Thread 1 (tenant-alpha): initially 0 prior memories, executes & persists resolution
        t1 = f"thr_mem_a1_{uuid.uuid4().hex[:8]}"
        async with (
            await psycopg.AsyncConnection.connect(settings.database_url) as conn,
            conn.transaction(),
        ):
            await set_tenant_context(conn, "tenant-alpha")
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
        ctx1 = RunContext(
            tenant_id="tenant-alpha",
            requester_id="alice",
            requester_roles=frozenset({"engineer"}),
            thread_id=t1,
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            request_id="req-mem-1",
        )
        cfg1 = {"configurable": {"thread_id": t1}}
        await graph.ainvoke(
            {"request_text": "checkout-api returning 502s after v2.4.1 deploy"},
            config=cfg1,
            context=ctx1,
        )
        snap1 = await graph.aget_state(cfg1)
        assert snap1.values["prior_memories"] == []
        assert snap1.interrupts
        intr = snap1.interrupts[0].value

        await graph.ainvoke(
            Command(
                resume={
                    "decision": "approve",
                    "approver_id": "bob",
                    "approved_plan_hash": intr["plan_hash"],
                }
            ),
            config=cfg1,
            context=ctx1,
        )
        end1 = await graph.aget_state(cfg1)
        assert end1.values["verification_passed"] is True
        op_id_1 = end1.values["operation_id"]

        # Thread 2 (tenant-alpha, NEW thread_id): recalls Thread 1's resolution memory
        t2 = f"thr_mem_a2_{uuid.uuid4().hex[:8]}"
        ctx2 = RunContext(
            tenant_id="tenant-alpha",
            requester_id="alice",
            requester_roles=frozenset({"engineer"}),
            thread_id=t2,
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            request_id="req-mem-2",
        )
        cfg2 = {"configurable": {"thread_id": t2}}
        await graph.ainvoke(
            {"request_text": "status only check on checkout-api"},
            config=cfg2,
            context=ctx2,
        )
        end2 = await graph.aget_state(cfg2)
        alpha_memories = end2.values["prior_memories"]
        assert len(alpha_memories) == 1
        assert alpha_memories[0]["operation_id"] == op_id_1
        assert alpha_memories[0]["thread_id"] == t1
        assert alpha_memories[0]["target"] == "checkout-api"
        assert alpha_memories[0]["action"] == "rollback"

        # Thread 3 (tenant-beta, NEW thread_id): sees ZERO memories from tenant-alpha
        t3 = f"thr_mem_b1_{uuid.uuid4().hex[:8]}"
        ctx3 = RunContext(
            tenant_id="tenant-beta",
            requester_id="carol",
            requester_roles=frozenset({"engineer"}),
            thread_id=t3,
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            request_id="req-mem-3",
        )
        cfg3 = {"configurable": {"thread_id": t3}}
        await graph.ainvoke(
            {"request_text": "status only check on billing-db"},
            config=cfg3,
            context=ctx3,
        )
        end3 = await graph.aget_state(cfg3)
        assert end3.values["prior_memories"] == []
