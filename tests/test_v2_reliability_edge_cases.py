"""Boundary-first regression tests for concurrent full-balance retries & timeout isolation.

Reproduces:
  1. Two simultaneous full-balance refunds with the same `operation_id` on PostgresDomainStore
     must serialize on `operation_id` so the retry returns deduplicated success (not
     `ValueError("Insufficient balance")`).
  2. A write timeout followed by a status-lookup timeout must NOT escape the graph; it must
     write an `outcome_unknown` reconciliation audit row and settle `outcome_unknown`.
  3. A post-write `verify_end_state` read-back timeout must NOT escape as `run_timeout`; it must
     preserve `execution_result` and mark `verification_passed=False, verification_failed`.
  4. A ticket-update timeout must audit `tool_id="update_ticket"` (resolved before execution).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.config import get_settings
from core.db import apply_app_migrations
from core.models import RunContext
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph
from workflows.refund_ops.graph import build_refund_ops_graph


def _ctx(thread_id: str) -> RunContext:
    return RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id=thread_id,
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        request_id="req-rel",
    )


async def test_postgres_full_balance_concurrent_refund_retry_returns_deduplicated_success() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute("TRUNCATE TABLE mock_operations RESTART IDENTITY CASCADE;")
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    store = PostgresDomainStore(settings.database_url)
    op_id = f"op_full_bal_{uuid.uuid4().hex[:8]}"

    # ACC-1001 starts with balance_cents = 100_000. Fire 4 simultaneous full-balance retries!
    results = await asyncio.gather(
        *(
            store.issue_refund(
                tenant_id="tenant-alpha",
                account_id="ACC-1001",
                amount_cents=100_000,
                reason="full balance refund retry",
                operation_id=op_id,
            )
            for _ in range(4)
        )
    )
    assert len([r for r in results if not r["deduplicated"]]) == 1
    assert len([r for r in results if r["deduplicated"]]) == 3
    acct = await store.get_account(tenant_id="tenant-alpha", account_id="ACC-1001")
    assert acct["balance_cents"] == 0


async def test_double_timeout_write_and_reconcile_records_outcome_unknown_audit() -> None:
    class _TimeoutReconcileStore(InMemoryDomainStore):
        async def get_operation_status(
            self, *, tenant_id: str, operation_id: str
        ) -> dict[str, Any]:
            raise TimeoutError("Simulated status lookup timeout")

    store = _TimeoutReconcileStore()
    deps = WorkflowDeps(
        domain_store=store,
        use_postgres=False,
        use_mcp_http=False,
        simulate_write_timeout=True,
    )
    graph = build_refund_ops_graph(deps=deps, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "rel-double-to"}}
    ctx = _ctx("rel-double-to")

    await graph.ainvoke({"request_text": "Issue $50 refund to ACC-1001"}, config=cfg, context=ctx)
    intr = (await graph.aget_state(cfg)).interrupts[0].value

    # Must NOT raise TimeoutError out of the graph!
    await graph.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": intr["plan_hash"],
            }
        ),
        config=cfg,
        context=ctx,
    )
    end = await graph.aget_state(cfg)
    assert end.values["final_status"] == "outcome_unknown"
    assert end.values["error_category"] == "outcome_unknown"
    rec_audits = [a for a in deps.audit_memory if a.event_type == "reconciliation"]
    assert len(rec_audits) == 1
    assert rec_audits[0].decision == "outcome_unknown"


async def test_post_write_verification_timeout_preserves_committed_write() -> None:
    class _TimeoutVerifyStore(InMemoryDomainStore):
        def __init__(self) -> None:
            super().__init__()
            self._reads = 0

        async def get_service_status(
            self, *, tenant_id: str, service_name: str
        ) -> dict[str, Any]:
            self._reads += 1
            # First read is during `investigate`; second read is during `verify_end_state`
            if self._reads >= 2:
                raise TimeoutError("Simulated post-write read-back timeout")
            return await super().get_service_status(
                tenant_id=tenant_id, service_name=service_name
            )

    store = _TimeoutVerifyStore()
    deps = WorkflowDeps(domain_store=store, use_postgres=False, use_mcp_http=False)
    graph = build_incident_desk_graph(deps=deps, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "rel-ver-to"}}
    ctx = _ctx("rel-ver-to")

    await graph.ainvoke({"request_text": "checkout-api returning 502s"}, config=cfg, context=ctx)
    intr = (await graph.aget_state(cfg)).interrupts[0].value

    # Must NOT raise TimeoutError; write committed, verification failed cleanly
    await graph.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": intr["plan_hash"],
            }
        ),
        config=cfg,
        context=ctx,
    )
    end = await graph.aget_state(cfg)
    assert end.values["execution_result"] is not None
    assert end.values["execution_result"]["current_version"] == "v2.4.0"
    assert end.values["verification_passed"] is False
    assert end.values["final_status"] == "verification_failed"


async def test_ticket_write_timeout_audits_update_ticket_tool_id() -> None:
    store = InMemoryDomainStore()
    deps = WorkflowDeps(
        domain_store=store,
        use_postgres=False,
        use_mcp_http=False,
        simulate_write_timeout=True,
    )
    graph = build_incident_desk_graph(deps=deps, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "rel-tkt-to"}}
    ctx = _ctx("rel-tkt-to")

    await graph.ainvoke({"request_text": "checkout-api returning 502s"}, config=cfg, context=ctx)
    await graph.ainvoke(
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
        config=cfg,
        context=ctx,
    )
    write_audits = [a for a in deps.audit_memory if a.event_type == "tool_write"]
    assert len(write_audits) == 1
    assert write_audits[0].tool_id == "update_ticket"
    assert write_audits[0].decision == "outcome_unknown"


async def test_incident_desk_agent_exercises_deploy_history_and_ticket_tools() -> None:
    """Exercise all 3 read-only tools (status, recent deploys, ticket lookup) via create_agent."""
    store = InMemoryDomainStore()
    deps = WorkflowDeps(domain_store=store, use_postgres=False, use_mcp_http=False)
    graph = build_incident_desk_graph(deps=deps, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "rel-all-tools"}}
    ctx = _ctx("rel-all-tools")

    await graph.ainvoke(
        {"request_text": "status only: check checkout-api deploy history and ticket INC-1001"},
        config=cfg,
        context=ctx,
    )
    end = await graph.aget_state(cfg)
    assert end.values["final_status"] == "no_action"
    tool_msgs = [m for m in end.values["messages"] if getattr(m, "type", None) == "tool"]
    tool_names = {getattr(m, "name", "") for m in tool_msgs}
    assert {"get_service_status", "list_recent_deploys", "get_ticket"}.issubset(tool_names)
