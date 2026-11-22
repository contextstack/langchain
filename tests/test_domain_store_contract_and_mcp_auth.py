"""Shared DomainStore contract tests (InMemory == Postgres) & MCP write-token binding."""

from __future__ import annotations

import asyncio
import socket
import uuid

import psycopg
import pytest
import uvicorn

from core.auth import mint_mcp_token
from core.config import get_settings
from core.db import apply_app_migrations
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.billing_server import create_billing_server
from mcp_servers.domain_store import (
    DomainStore,
    IdempotencyConflictError,
    InMemoryDomainStore,
    PostgresDomainStore,
)
from mcp_servers.infra_server import create_infra_server
from mcp_servers.ticketing_server import create_ticketing_server
from workflows.mcp_client import call_mcp_tool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.parametrize("backend", ["memory", "postgres"])
async def test_domain_store_contract_parity(backend: str) -> None:
    """Both InMemoryDomainStore and PostgresDomainStore must honor identical semantics."""
    store: DomainStore
    if backend == "memory":
        store = InMemoryDomainStore()
    else:
        settings = get_settings()
        await apply_app_migrations(settings.database_admin_url)
        async with await psycopg.AsyncConnection.connect(
            settings.database_admin_url, autocommit=True
        ) as conn:
            await conn.execute("TRUNCATE TABLE mock_operations RESTART IDENTITY CASCADE;")
        async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
            await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())
        store = PostgresDomainStore(settings.database_url)

    # Parity check: rollback WITHOUT target_version marks service healthy in BOTH stores
    op_rb = f"op_rb_{uuid.uuid4().hex[:8]}"
    res = await store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version=None,
        operation_id=op_rb,
    )
    assert res["status"] == "healthy"
    assert res["current_version"] == "v2.4.1"
    assert res["deduplicated"] is False

    # Replay returns deduplicated=True; payload mismatch raises IdempotencyConflictError
    replay = await store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version=None,
        operation_id=op_rb,
    )
    assert replay["deduplicated"] is True
    with pytest.raises(IdempotencyConflictError):
        await store.execute_service_change(
            tenant_id="tenant-alpha",
            service_name="checkout-api",
            action="restart",
            target_version=None,
            operation_id=op_rb,
        )


async def test_postgres_domain_store_simultaneous_retries_same_operation_id() -> None:
    """5 concurrent identical issue_refund calls with the same operation_id deduct balance ONCE."""
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute("TRUNCATE TABLE mock_operations RESTART IDENTITY CASCADE;")
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    store = PostgresDomainStore(settings.database_url)
    op_id = f"op_race_{uuid.uuid4().hex[:8]}"

    results = await asyncio.gather(
        *(
            store.issue_refund(
                tenant_id="tenant-alpha",
                account_id="ACC-1001",
                amount_cents=10_000,
                reason="concurrent retry",
                operation_id=op_id,
            )
            for _ in range(5)
        )
    )
    fresh = [r for r in results if not r["deduplicated"]]
    deduped = [r for r in results if r["deduplicated"]]
    assert len(fresh) == 1
    assert len(deduped) == 4
    acct = await store.get_account(tenant_id="tenant-alpha", account_id="ACC-1001")
    assert acct["balance_cents"] == 90_000


async def test_mcp_write_handlers_reject_mismatched_signed_operation_id() -> None:
    """All 3 FastMCP write servers must reject idempotency_key != signed claims.operation_id."""
    secret = "mcp-binding-secret-key-at-least-32-bytes-long"
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

        signed_tok = mint_mcp_token(
            tenant_id="tenant-alpha",
            requester_id="alice",
            run_id="r1",
            approved_by="bob",
            operation_id="op_signed_1",
            secret=secret,
        )

        # Mismatched idempotency_key must fail on all 3 servers with zero mutations
        with pytest.raises(Exception):  # noqa: B017
            await call_mcp_tool(
                url=f"http://127.0.0.1:{p_infra}/mcp",
                bearer_token=signed_tok,
                tool_name="execute_service_change",
                arguments={
                    "service_name": "checkout-api",
                    "action": "restart",
                    "idempotency_key": "op_different",
                },
            )
        with pytest.raises(Exception):  # noqa: B017
            await call_mcp_tool(
                url=f"http://127.0.0.1:{p_ticket}/mcp",
                bearer_token=signed_tok,
                tool_name="update_ticket",
                arguments={
                    "ticket_id": "INC-1001",
                    "status": "resolved",
                    "note": "spoof",
                    "idempotency_key": "op_different",
                },
            )
        with pytest.raises(Exception):  # noqa: B017
            await call_mcp_tool(
                url=f"http://127.0.0.1:{p_bill}/mcp",
                bearer_token=signed_tok,
                tool_name="issue_refund",
                arguments={
                    "account_id": "ACC-1001",
                    "amount_cents": 100,
                    "reason": "spoof",
                    "idempotency_key": "op_different",
                },
            )
        assert len(store.operations) == 0
    finally:
        for s in srvs:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_service_version_normalization_and_validation_at_write_boundary() -> None:
    """DomainStore canonicalizes bare semver ('2.4.0' -> 'v2.4.0') and rejects invalid strings."""
    from mcp_servers.domain_store import normalize_service_version  # noqa: PLC0415

    assert normalize_service_version(None) is None
    assert normalize_service_version("  ") is None
    assert normalize_service_version("2.4.0") == "v2.4.0"
    assert normalize_service_version("v2.4.0") == "v2.4.0"
    with pytest.raises(ValueError, match="Invalid target_version"):
        normalize_service_version("latest")

    store = InMemoryDomainStore()
    res = await store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version="2.4.0",
        operation_id="op_norm_v1",
    )
    assert res["current_version"] == "v2.4.0"
    svc = await store.get_service_status(tenant_id="tenant-alpha", service_name="checkout-api")
    assert svc["current_version"] == "v2.4.0"

    with pytest.raises(ValueError, match="Invalid target_version"):
        await store.execute_service_change(
            tenant_id="tenant-alpha",
            service_name="checkout-api",
            action="rollback",
            target_version="not-a-version",
            operation_id="op_bad_v1",
        )
