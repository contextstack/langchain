"""M1 Tests: Core contracts, PolicyEngine, JWT/MCP auth, Idempotency, ACL RAG, & Audit DB."""

from __future__ import annotations

import asyncio

import psycopg
import pytest

from core.audit import AuditEvent, compute_audit_event_id, write_audit_event
from core.auth import (
    AuthError,
    mint_mcp_token,
    mint_user_jwt,
    verify_mcp_token,
    verify_user_jwt,
)
from core.config import get_settings
from core.db import apply_app_migrations, set_tenant_context
from core.models import CallerIdentity, ChangePlan, RunbookCitation, RunContext, ToolContract
from core.pii import redact_obj, redact_text
from core.policy import PolicyDecision, PolicyEngine
from core.tool_contract import TRUNCATION_MARKER, cap_tool_output, invoke_with_contract
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import retrieve_runbooks, seed_knowledge_and_domain
from mcp_servers.auth_middleware import extract_bearer_token
from mcp_servers.domain_store import (
    IdempotencyConflictError,
    InMemoryDomainStore,
    PostgresDomainStore,
    ResourceNotFoundError,
)


# ---------------------------------------------------------------------------
# 1. Auth & Forged/Expired/Missing Token Refusal
# ---------------------------------------------------------------------------
def test_user_and_mcp_jwt_roundtrip_and_forgery_rejection() -> None:
    alice = CallerIdentity(
        tenant_id="tenant-alpha",
        actor_id="alice",
        roles=frozenset({"engineer"}),
    )
    secret_a = "test-user-jwt-secret-key-at-least-32-bytes"
    wrong_secret = "wrong-user-jwt-secret-key-at-least-32-bytes"
    mcp_secret = "test-mcp-jwt-secret-key-at-least-32-bytes!"

    token = mint_user_jwt(alice, secret=secret_a)
    verified = verify_user_jwt(token, secret=secret_a)
    assert verified == alice

    # Wrong secret -> rejected
    with pytest.raises(AuthError):
        verify_user_jwt(token, secret=wrong_secret)

    # Wrong audience (user JWT passed to MCP verifier) -> rejected
    with pytest.raises(AuthError):
        verify_mcp_token(token, secret=secret_a)

    # Valid MCP write token
    mcp_tok = mint_mcp_token(
        tenant_id="tenant-alpha",
        requester_id="alice",
        run_id="run-1",
        approved_by="bob",
        operation_id="op-123",
        secret=mcp_secret,
    )
    claims = verify_mcp_token(mcp_tok, secret=mcp_secret)
    assert claims.tenant_id == "tenant-alpha"
    assert claims.approved_by == "bob"
    assert claims.operation_id == "op-123"

    # Expired MCP token -> rejected
    expired_tok = mint_mcp_token(
        tenant_id="tenant-alpha",
        requester_id="alice",
        run_id="run-1",
        secret=mcp_secret,
        ttl_s=-10,
    )
    with pytest.raises(AuthError):
        verify_mcp_token(expired_tok, secret=mcp_secret)

    # Missing Bearer header -> rejected
    with pytest.raises(AuthError):
        extract_bearer_token({})


# ---------------------------------------------------------------------------
# 2. PolicyEngine & Separation of Duties
# ---------------------------------------------------------------------------
def test_policy_engine_and_separation_of_duties() -> None:
    engine = PolicyEngine()
    ctx = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="t-1",
        run_id="r-1",
        request_id="req-1",
    )

    # Normal write requires approval (no write auto-allowed in v1)
    plan_ok = ChangePlan(
        resource_type="service",
        target_resource="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        citations=[RunbookCitation(doc_id="rb-alpha-checkout-502", title="Rollback")],
    )
    assert engine.evaluate_change_plan(ctx=ctx, plan=plan_ok).decision == (
        PolicyDecision.REQUIRE_APPROVAL
    )

    # Disallowed destructive action denied
    plan_bad = ChangePlan(
        resource_type="service",
        target_resource="checkout-api",
        action="drop_database",
    )
    assert engine.evaluate_change_plan(ctx=ctx, plan=plan_bad).decision == PolicyDecision.DENY

    # Cross-tenant target smuggled in resource name denied
    plan_cross = ChangePlan(
        resource_type="service",
        target_resource="tenant-beta/billing-db",
        action="restart",
    )
    assert engine.evaluate_change_plan(ctx=ctx, plan=plan_cross).decision == PolicyDecision.DENY

    # Separation of duties: dana has both roles, cannot approve her own request
    dana = CallerIdentity(
        tenant_id="tenant-alpha",
        actor_id="dana",
        roles=frozenset({"engineer", "ops-approver"}),
    )
    self_res = engine.check_approval_authority(
        caller=dana,
        approval_tenant_id="tenant-alpha",
        requested_by="dana",
    )
    assert self_res.decision == PolicyDecision.DENY

    # Bob (same tenant, ops-approver, different person) is authorized
    bob = CallerIdentity(
        tenant_id="tenant-alpha",
        actor_id="bob",
        roles=frozenset({"ops-approver"}),
    )
    ok_res = engine.check_approval_authority(
        caller=bob,
        approval_tenant_id="tenant-alpha",
        requested_by="dana",
    )
    assert ok_res.decision == PolicyDecision.ALLOW

    # Cross-tenant approver denied
    carol_beta = CallerIdentity(
        tenant_id="tenant-beta",
        actor_id="carol",
        roles=frozenset({"ops-approver"}),
    )
    cross_res = engine.check_approval_authority(
        caller=carol_beta,
        approval_tenant_id="tenant-alpha",
        requested_by="dana",
    )
    assert cross_res.decision == PolicyDecision.DENY


# ---------------------------------------------------------------------------
# 3. PII Redaction & ChangePlan Hashing
# ---------------------------------------------------------------------------
def test_pii_redaction_and_plan_hash_determinism() -> None:
    raw = "Contact alice@example.com or 555-867-5309 with key sk-proj-1234567890abcdef"
    masked = redact_text(raw)
    assert "alice@example.com" not in masked
    assert "555-867-5309" not in masked
    assert "sk-proj-1234567890abcdef" not in masked
    assert "[REDACTED_EMAIL]" in masked
    assert "[REDACTED_PHONE]" in masked
    assert "[REDACTED_API_KEY]" in masked

    nested = redact_obj({"user": {"email": "bob@corp.io"}, "items": ["sk-test-abcdefghijklmnop"]})
    assert nested["user"]["email"] == "[REDACTED_EMAIL]"
    assert nested["items"][0] == "[REDACTED_API_KEY]"

    p1 = ChangePlan(resource_type="service", target_resource="checkout-api", action="rollback")
    p2 = ChangePlan(resource_type="service", target_resource="checkout-api", action="rollback")
    p3 = ChangePlan(resource_type="service", target_resource="checkout-api", action="restart")
    assert p1.plan_hash() == p2.plan_hash()
    assert p1.plan_hash() != p3.plan_hash()


# ---------------------------------------------------------------------------
# 4. Audit UUIDv5 Idempotency Without Collapsing Distinct Outcomes
# ---------------------------------------------------------------------------
def test_audit_event_id_deduplicates_replays_but_preserves_distinct_outcomes() -> None:
    id_a1 = compute_audit_event_id(
        thread_id="t-1",
        event_type="tool_write",
        step_or_op_key="op-1",
        attempt=1,
        decision="outcome_unknown",
    )
    id_a1_replay = compute_audit_event_id(
        thread_id="t-1",
        event_type="tool_write",
        step_or_op_key="op-1",
        attempt=1,
        decision="outcome_unknown",
    )
    id_a2_resolved = compute_audit_event_id(
        thread_id="t-1",
        event_type="tool_write",
        step_or_op_key="op-1",
        attempt=2,
        decision="succeeded",
    )
    assert id_a1 == id_a1_replay
    assert id_a1 != id_a2_resolved


# ---------------------------------------------------------------------------
# 5. Tool Contract Wrapper (Output Cap & Timeout)
# ---------------------------------------------------------------------------
async def test_tool_contract_caps_output_and_enforces_timeout() -> None:
    long_text = "x" * 200
    capped = cap_tool_output(long_text, max_chars=80)
    assert len(capped) <= 80
    assert capped.endswith(TRUNCATION_MARKER)

    fast_contract = ToolContract(
        tool_id="slow_tool",
        read_write_class="read",
        timeout_s=0.05,
        max_output_chars=100,
    )

    async def slow_fn() -> str:
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await invoke_with_contract(fast_contract, slow_fn)


# ---------------------------------------------------------------------------
# 6. Domain Store Idempotency, Conflict Detection & Tenant Isolation
# ---------------------------------------------------------------------------
async def test_inmemory_domain_store_idempotency_and_tenant_isolation() -> None:
    store = InMemoryDomainStore()

    # Tenant isolation: tenant-alpha cannot read tenant-beta's billing-db
    with pytest.raises(ResourceNotFoundError):
        await store.get_service_status(tenant_id="tenant-alpha", service_name="billing-db")

    # First write commits
    res1 = await store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        operation_id="op-rollback-1",
    )
    assert res1["deduplicated"] is False
    assert res1["current_version"] == "v2.4.0"
    assert res1["status"] == "healthy"

    # Repeat with SAME operation_id and SAME payload -> deduplicated
    res2 = await store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        operation_id="op-rollback-1",
    )
    assert res2["deduplicated"] is True
    assert len(store.operations) == 1

    # Reuse SAME operation_id with DIFFERENT payload -> IdempotencyConflictError
    with pytest.raises(IdempotencyConflictError):
        await store.execute_service_change(
            tenant_id="tenant-alpha",
            service_name="checkout-api",
            action="restart",
            operation_id="op-rollback-1",
        )

    # Reconciliation is tenant-scoped
    op_alpha = await store.get_operation_status(
        tenant_id="tenant-alpha", operation_id="op-rollback-1"
    )
    op_beta = await store.get_operation_status(
        tenant_id="tenant-beta", operation_id="op-rollback-1"
    )
    assert op_alpha["status"] == "succeeded"
    assert op_beta["status"] == "not_found"


# ---------------------------------------------------------------------------
# 7. Live Postgres Integration Tests (AC-12, pgvector ACLs, PostgresDomainStore)
#    Auto-skips cleanly if Docker Postgres is not running locally.
# ---------------------------------------------------------------------------
async def _postgres_available() -> bool:
    settings = get_settings()
    try:
        async with await psycopg.AsyncConnection.connect(
            settings.database_admin_url, connect_timeout=1
        ):
            return True
    except Exception:
        return False


async def test_postgres_migrations_ac12_append_only_and_acl_rag() -> None:
    if not await _postgres_available():
        pytest.skip("Local Postgres (docker compose up -d) is not running")

    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)

    # Seed using app_rw
    embeddings = DeterministicMockEmbeddings()
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        counts = await seed_knowledge_and_domain(conn, embeddings=embeddings)
        assert counts["runbooks"] == 5

        # ACL RAG: tenant-alpha engineer never sees tenant-beta OR approver-only docs
        docs_alpha_eng = await retrieve_runbooks(
            conn,
            embeddings=embeddings,
            tenant_id="tenant-alpha",
            roles=frozenset({"engineer"}),
            query="billing-db failover secret checkout-api",
            limit=10,
        )
        doc_ids = {d.doc_id for d in docs_alpha_eng}
        assert "rb-alpha-checkout-502" in doc_ids
        assert "rb-beta-billing-secret" not in doc_ids
        assert "rb-alpha-approver-secrets" not in doc_ids

        # AC-12: audit_log INSERT succeeds, UPDATE and DELETE fail under app_rw
        ev = AuditEvent(
            tenant_id="tenant-alpha",
            actor_id="alice",
            thread_id="t-ac12",
            run_id="r-ac12",
            workflow_version="0.1.0+test",
            event_type="authz_check",
            decision="allow",
            step_or_op_key="ac12-check",
        )
        async with conn.transaction():
            await set_tenant_context(conn, "tenant-alpha")
            event_id = await write_audit_event(conn, ev)

        with pytest.raises(psycopg.Error):
            async with conn.transaction():
                await set_tenant_context(conn, "tenant-alpha")
                await conn.execute(
                    "UPDATE audit_log SET decision = 'deny' WHERE event_id = %s",
                    (event_id,),
                )

        with pytest.raises(psycopg.Error):
            async with conn.transaction():
                await set_tenant_context(conn, "tenant-alpha")
                await conn.execute(
                    "DELETE FROM audit_log WHERE event_id = %s",
                    (event_id,),
                )

    # Live PostgresDomainStore idempotency check
    pg_store = PostgresDomainStore(settings.database_url)
    op_id = "op-pg-test-rollback-1"
    r1 = await pg_store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        operation_id=op_id,
    )
    r2 = await pg_store.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        operation_id=op_id,
    )
    assert r1["current_version"] == "v2.4.0"
    assert r2["deduplicated"] is True
