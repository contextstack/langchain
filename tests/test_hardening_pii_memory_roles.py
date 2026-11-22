"""Boundary tests for PII redaction, memory prompt injection, api_rw RLS, and runtime status."""

from __future__ import annotations

import psycopg
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from api.main import create_app
from core.config import Settings, get_settings
from core.db import apply_app_migrations, set_system_rls_bypass, set_tenant_context
from core.models import RunContext
from core.pii import redact_text
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import InMemoryDomainStore
from workflows.incident_desk.graph import build_incident_desk_graph
from workflows.lifecycle import WorkflowDeps


def test_broadened_pii_detectors_mask_secrets_without_false_positives() -> None:
    jwt_tok = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    slack_tok = "xoxb-" + ("0" * 12) + "-" + ("0" * 12) + "-" + ("a" * 24)
    raw = (
        f"Contact alice@example.com or 4155551234 (SSN 123-45-6789, "
        f"card 4111-2222-3333-4444), jwt={jwt_tok}, slack={slack_tok}. "
        f"Keep ACC-1001 balance 100000 and op_0123456789abcdef intact."
    )
    masked = redact_text(raw)
    assert "alice@example.com" not in masked and "[REDACTED_EMAIL]" in masked
    assert "4155551234" not in masked and "[REDACTED_PHONE]" in masked
    assert "123-45-6789" not in masked and "[REDACTED_SSN]" in masked
    assert "4111-2222-3333-4444" not in masked and "[REDACTED_CARD]" in masked
    assert jwt_tok not in masked and "[REDACTED_JWT]" in masked
    assert slack_tok not in masked and "[REDACTED_SLACK_TOKEN]" in masked
    # No false positives on domain identifiers or balances
    assert "ACC-1001" in masked
    assert "100000" in masked
    assert "op_0123456789abcdef" in masked


async def test_cross_thread_memory_injected_into_investigate_prompt() -> None:
    store = InMemoryStore()
    await store.aput(
        ("tenant-alpha", "memory", "incident_desk"),
        "mem-1",
        {"target": "checkout-api", "action": "rollback", "summary": "Rolled back v2.4.1 -> v2.4.0"},
    )
    deps = WorkflowDeps(domain_store=InMemoryDomainStore(), use_postgres=False)
    graph = build_incident_desk_graph(deps=deps, checkpointer=InMemorySaver(), store=store)
    ctx = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="t-mem-prompt",
        run_id="r-mem-prompt",
        request_id="req-mem-prompt",
    )
    final = await graph.ainvoke(
        {"request_text": "status only check on checkout-api"},
        config={"configurable": {"thread_id": ctx.thread_id}},
        context=ctx,
        durability="sync",
    )
    assert final["final_status"] == "no_action"
    first_msg = str(getattr(final["messages"][0], "content", ""))
    assert "Prior Tenant Memories" in first_msg
    assert "Rolled back v2.4.1 -> v2.4.0" in first_msg


async def test_api_rw_role_cannot_bypass_rls() -> None:
    """Real PostgreSQL boundary check: api_rw ignores app.bypass_rls='on'."""
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    api_dsn = settings.database_url.replace(
        "app_rw:app_rw_dev_password", "api_rw:api_rw_dev_password"
    )
    async with await psycopg.AsyncConnection.connect(api_dsn) as conn:
        # Attempt system RLS bypass as api_rw -> must STILL return 0 rows!
        async with conn.transaction(), conn.cursor() as cur:
            await set_system_rls_bypass(conn)
            await cur.execute("SELECT COUNT(*) FROM mock_services")
            row = await cur.fetchone()
            assert row is not None and row[0] == 0

        # With explicit tenant context -> returns ONLY tenant-alpha rows
        async with conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, "tenant-alpha")
            await cur.execute("SELECT DISTINCT tenant_id FROM mock_services")
            tenants = [r[0] for r in await cur.fetchall()]
            assert tenants == ["tenant-alpha"]


def test_runtime_status_endpoint_exposes_non_sensitive_posture() -> None:
    cfg = Settings(
        model_mode="mock",
        mcp_transport="direct",
        telemetry_backend="none",
        api_database_url="postgresql://api_rw:api_rw_dev_password@127.0.0.1:5439/contextstack",
    )
    app = create_app(settings=cfg)
    with TestClient(app) as client:
        resp = client.get("/v1/runtime/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_mode"] == "mock"
        assert body["mcp_transport"] == "direct"
        assert body["telemetry_backend"] == "none"
        assert body["db_role_isolated"] is True
        assert body["workflows"] == ["incident_desk", "refund_ops"]
