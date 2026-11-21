"""Targeted tests for core policy/contracts, domain stores, FastMCP servers, and hardening."""

from __future__ import annotations

import asyncio
import socket

import httpx
import psycopg
import pytest
import uvicorn

from api.main import create_app
from core.auth import mint_mcp_token
from core.config import Settings, get_settings
from core.db import apply_app_migrations
from core.models import ChangePlan, RunContext, ToolContract
from core.policy import PolicyDecision, PolicyEngine
from core.tool_contract import TRUNCATION_MARKER, invoke_with_contract
from gateway.factory import get_embeddings
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import retrieve_runbooks, seed_knowledge_and_domain
from mcp_servers.billing_server import create_billing_server
from mcp_servers.domain_store import (
    IdempotencyConflictError,
    InMemoryDomainStore,
    PostgresDomainStore,
    ResourceNotFoundError,
)
from mcp_servers.infra_server import create_infra_server
from mcp_servers.ticketing_server import create_ticketing_server
from workflows.mcp_client import MCPToolCallError, _parse_mcp_result, call_mcp_tool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_policy_engine_tool_and_refund_branches() -> None:
    engine = PolicyEngine()
    ctx = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="t1",
        run_id="r1",
        request_id="req1",
    )
    read_contract = ToolContract(
        tool_id="get_service_status", version="1.0.0", read_write_class="read"
    )
    write_contract = ToolContract(
        tool_id="issue_refund",
        version="1.0.0",
        read_write_class="write",
        idempotency_required=True,
        audit_required=True,
    )

    assert (
        engine.evaluate_tool_call(ctx=ctx, contract=read_contract).decision
        == PolicyDecision.ALLOW
    )
    assert (
        engine.evaluate_tool_call(
            ctx=ctx, contract=read_contract, target_tenant_id="tenant-beta"
        ).decision
        == PolicyDecision.DENY
    )
    assert (
        engine.evaluate_tool_call(ctx=ctx, contract=write_contract).decision
        == PolicyDecision.REQUIRE_APPROVAL
    )

    # Unauthorized requester role
    no_role_ctx = RunContext(
        tenant_id="tenant-alpha",
        requester_id="viewer",
        requester_roles=frozenset({"readonly"}),
        thread_id="t1",
        run_id="r1",
        request_id="req1",
    )
    plan_ok = ChangePlan(
        resource_type="account",
        target_resource="ACC-1001",
        action="issue_refund",
        amount_cents=10_000,
        risk_level="medium",
        summary="Refund $100",
    )
    assert (
        engine.evaluate_change_plan(ctx=no_role_ctx, plan=plan_ok).decision
        == PolicyDecision.DENY
    )

    # No-action plan
    plan_none = ChangePlan(
        resource_type="none", target_resource="", action="none", risk_level="low", summary="noop"
    )
    assert engine.evaluate_change_plan(ctx=ctx, plan=plan_none).decision == PolicyDecision.ALLOW

    # Refund policy branches: valid, non-positive, over ceiling
    assert (
        engine.evaluate_change_plan(ctx=ctx, plan=plan_ok).decision
        == PolicyDecision.REQUIRE_APPROVAL
    )
    plan_zero = plan_ok.model_copy(update={"amount_cents": 0})
    assert engine.evaluate_change_plan(ctx=ctx, plan=plan_zero).decision == PolicyDecision.DENY
    plan_huge = plan_ok.model_copy(update={"amount_cents": 600_000})
    assert engine.evaluate_change_plan(ctx=ctx, plan=plan_huge).decision == PolicyDecision.DENY


async def test_invoke_with_contract_string_and_non_string() -> None:
    contract = ToolContract(
        tool_id="demo", version="1.0.0", read_write_class="read", max_output_chars=40
    )

    async def _long_str() -> str:
        return "x" * 100

    async def _dict_val() -> dict[str, int]:
        return {"ok": 1}

    out_str = await invoke_with_contract(contract, _long_str)
    assert out_str.endswith(TRUNCATION_MARKER)
    assert await invoke_with_contract(contract, _dict_val) == {"ok": 1}


async def test_inmemory_domain_store_tickets_accounts_and_refunds() -> None:
    store = InMemoryDomainStore()
    missing_op = await store.get_operation_status(
        tenant_id="tenant-alpha", operation_id="op_missing"
    )
    assert missing_op == {
        "operation_id": "op_missing",
        "status": "not_found",
    }
    deploys = await store.list_recent_deploys(tenant_id="tenant-alpha", service_name="checkout-api")
    assert deploys[0]["previous_version"] == "v2.4.0"

    # Tickets: read, update, dedup, conflict, not found
    t = await store.get_ticket(tenant_id="tenant-alpha", ticket_id="INC-1001")
    assert t["status"] == "open"
    with pytest.raises(ResourceNotFoundError):
        await store.get_ticket(tenant_id="tenant-alpha", ticket_id="INC-9999")
    with pytest.raises(ResourceNotFoundError):
        await store.update_ticket(
            tenant_id="tenant-alpha",
            ticket_id="INC-9999",
            status="resolved",
            note="n",
            operation_id="op_t_nf",
        )
    u1 = await store.update_ticket(
        tenant_id="tenant-alpha",
        ticket_id="INC-1001",
        status="resolved",
        note="fixed",
        operation_id="op_t1",
    )
    assert u1["deduplicated"] is False
    u2 = await store.update_ticket(
        tenant_id="tenant-alpha",
        ticket_id="INC-1001",
        status="resolved",
        note="fixed",
        operation_id="op_t1",
    )
    assert u2["deduplicated"] is True
    with pytest.raises(IdempotencyConflictError):
        await store.update_ticket(
            tenant_id="tenant-alpha",
            ticket_id="INC-1001",
            status="resolved",
            note="different",
            operation_id="op_t1",
        )
    op_stat = await store.get_operation_status(tenant_id="tenant-alpha", operation_id="op_t1")
    assert op_stat["status"] == "succeeded"

    # Accounts & refunds: read, not found, validation, happy path, dedup, conflict
    acct = await store.get_account(tenant_id="tenant-alpha", account_id="ACC-1001")
    assert acct["balance_cents"] == 100_000
    with pytest.raises(ResourceNotFoundError):
        await store.get_account(tenant_id="tenant-alpha", account_id="ACC-9999")
    with pytest.raises(ResourceNotFoundError):
        await store.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-9999",
            amount_cents=100,
            reason="r",
            operation_id="op_r_nf",
        )
    with pytest.raises(ValueError, match="positive"):
        await store.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-1001",
            amount_cents=0,
            reason="r",
            operation_id="op_r_zero",
        )
    with pytest.raises(ValueError, match="Insufficient"):
        await store.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-1001",
            amount_cents=999_999,
            reason="r",
            operation_id="op_r_big",
        )
    r1 = await store.issue_refund(
        tenant_id="tenant-alpha",
        account_id="ACC-1001",
        amount_cents=5_000,
        reason="sla",
        operation_id="op_r1",
    )
    assert r1["new_balance_cents"] == 95_000
    assert r1["deduplicated"] is False
    r2 = await store.issue_refund(
        tenant_id="tenant-alpha",
        account_id="ACC-1001",
        amount_cents=5_000,
        reason="sla",
        operation_id="op_r1",
    )
    assert r2["deduplicated"] is True
    with pytest.raises(IdempotencyConflictError):
        await store.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-1001",
            amount_cents=6_000,
            reason="sla",
            operation_id="op_r1",
        )


async def test_postgres_domain_store_tickets_and_refund_edges() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute("TRUNCATE TABLE mock_operations RESTART IDENTITY CASCADE;")
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    pg = PostgresDomainStore(settings.database_url)
    deploys = await pg.list_recent_deploys(tenant_id="tenant-alpha", service_name="checkout-api")
    assert deploys[0]["service_name"] == "checkout-api"

    t = await pg.get_ticket(tenant_id="tenant-alpha", ticket_id="INC-1001")
    assert t["ticket_id"] == "INC-1001"
    with pytest.raises(ResourceNotFoundError):
        await pg.get_ticket(tenant_id="tenant-alpha", ticket_id="INC-NOPE")
    with pytest.raises(ResourceNotFoundError):
        await pg.update_ticket(
            tenant_id="tenant-alpha",
            ticket_id="INC-NOPE",
            status="resolved",
            note="x",
            operation_id="op_pg_t_nf",
        )
    u1 = await pg.update_ticket(
        tenant_id="tenant-alpha",
        ticket_id="INC-1001",
        status="resolved",
        note="pg fixed",
        operation_id="op_pg_t1",
    )
    assert u1["deduplicated"] is False
    u2 = await pg.update_ticket(
        tenant_id="tenant-alpha",
        ticket_id="INC-1001",
        status="resolved",
        note="pg fixed",
        operation_id="op_pg_t1",
    )
    assert u2["deduplicated"] is True
    with pytest.raises(IdempotencyConflictError):
        await pg.update_ticket(
            tenant_id="tenant-alpha",
            ticket_id="INC-1001",
            status="resolved",
            note="tampered",
            operation_id="op_pg_t1",
        )

    # Postgres refund validation & conflict paths
    with pytest.raises(ResourceNotFoundError):
        await pg.get_account(tenant_id="tenant-alpha", account_id="ACC-NOPE")
    with pytest.raises(ValueError, match="positive"):
        await pg.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-1001",
            amount_cents=-10,
            reason="bad",
            operation_id="op_pg_neg",
        )
    with pytest.raises(ValueError, match="Insufficient"):
        await pg.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-1001",
            amount_cents=999_999,
            reason="too big",
            operation_id="op_pg_big",
        )
    with pytest.raises(ResourceNotFoundError):
        await pg.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-NOPE",
            amount_cents=100,
            reason="nf",
            operation_id="op_pg_r_nf",
        )
    r1 = await pg.issue_refund(
        tenant_id="tenant-alpha",
        account_id="ACC-1001",
        amount_cents=1_000,
        reason="ok",
        operation_id="op_pg_r1",
    )
    assert r1["deduplicated"] is False
    with pytest.raises(IdempotencyConflictError):
        await pg.issue_refund(
            tenant_id="tenant-alpha",
            account_id="ACC-1001",
            amount_cents=2_000,
            reason="ok",
            operation_id="op_pg_r1",
        )


async def test_live_mcp_servers_ticketing_infra_billing_and_client_edges() -> None:
    assert _parse_mcp_result({"ok": True}) == {"ok": True}
    with pytest.raises(MCPToolCallError):
        _parse_mcp_result("plain error string")

    secret = "mcp-test-secret-key-at-least-32-bytes-long"
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

        read_tok = mint_mcp_token(
            tenant_id="tenant-alpha", requester_id="alice", run_id="r1", secret=secret
        )
        write_tok = mint_mcp_token(
            tenant_id="tenant-alpha",
            requester_id="alice",
            run_id="r1",
            approved_by="bob",
            operation_id="op_1",
            secret=secret,
        )

        # Infra list_recent_deploys + unknown tool error
        deploys = await call_mcp_tool(
            url=f"http://127.0.0.1:{p_infra}/mcp",
            bearer_token=read_tok,
            tool_name="list_recent_deploys",
            arguments={"service_name": "checkout-api"},
        )
        assert deploys[0]["service_name"] == "checkout-api"
        with pytest.raises(MCPToolCallError, match="not found"):
            await call_mcp_tool(
                url=f"http://127.0.0.1:{p_infra}/mcp",
                bearer_token=read_tok,
                tool_name="nonexistent_tool",
                arguments={},
            )

        # Ticketing read + unapproved write refusal + approved write
        t = await call_mcp_tool(
            url=f"http://127.0.0.1:{p_ticket}/mcp",
            bearer_token=read_tok,
            tool_name="get_ticket",
            arguments={"ticket_id": "INC-1001"},
        )
        assert t["ticket_id"] == "INC-1001"
        with pytest.raises(Exception):  # noqa: B017
            await call_mcp_tool(
                url=f"http://127.0.0.1:{p_ticket}/mcp",
                bearer_token=read_tok,
                tool_name="update_ticket",
                arguments={
                    "ticket_id": "INC-1001",
                    "status": "resolved",
                    "note": "nope",
                    "idempotency_key": "op_unauth",
                },
            )
        upd = await call_mcp_tool(
            url=f"http://127.0.0.1:{p_ticket}/mcp",
            bearer_token=write_tok,
            tool_name="update_ticket",
            arguments={
                "ticket_id": "INC-1001",
                "status": "resolved",
                "note": "approved note",
                "idempotency_key": "op_1",
            },
        )
        assert upd["status"] == "resolved"

        # Billing unapproved write refusal
        with pytest.raises(Exception):  # noqa: B017
            await call_mcp_tool(
                url=f"http://127.0.0.1:{p_bill}/mcp",
                bearer_token=read_tok,
                tool_name="issue_refund",
                arguments={
                    "account_id": "ACC-1001",
                    "amount_cents": 100,
                    "reason": "nope",
                    "idempotency_key": "op_b_unauth",
                },
            )
    finally:
        for s in srvs:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_embedding_query_pii_redaction_tenant_key_and_dev_token_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Retriever redacts PII before calling embed_query
    seen_queries: list[str] = []

    class _CapturingEmbeddings(DeterministicMockEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            seen_queries.append(text)
            return super().embed_query(text)

    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await retrieve_runbooks(
            conn,
            embeddings=_CapturingEmbeddings(),
            tenant_id="tenant-alpha",
            roles=frozenset({"engineer"}),
            query="user alice@corp.example called 415-555-0199 with sk-live-9876543210abcdef",
        )
    assert len(seen_queries) == 1
    assert "alice@corp.example" not in seen_queries[0]
    assert "415-555-0199" not in seen_queries[0]
    assert "sk-live-9876543210abcdef" not in seen_queries[0]

    # 2. get_embeddings uses tenant-scoped key in gateway mode
    monkeypatch.setenv("LITELLM_KEY_TENANT_ALPHA", "sk-alpha-virtual-key")
    gw_settings = Settings(model_mode="gateway")
    emb = get_embeddings(tenant_id="tenant-alpha", settings=gw_settings)
    assert emb.openai_api_key.get_secret_value() == "sk-alpha-virtual-key"

    # 3. /v1/dev/token returns 403 when enable_dev_tokens=False
    disabled_app = create_app(Settings(enable_dev_tokens=False))
    async with (
        disabled_app.router.lifespan_context(disabled_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=disabled_app), base_url="http://test"
        ) as client,
    ):
        resp = await client.post("/v1/dev/token", json={"persona": "alice"})
        assert resp.status_code == 403
