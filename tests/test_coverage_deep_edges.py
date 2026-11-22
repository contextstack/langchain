"""Deep edge-case tests pushing testable logic across API, auth, telemetry, judge, and workflows."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import jwt
import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from api.main import create_app
from core.auth import (
    API_AUDIENCE,
    JWT_ALGORITHM,
    MCP_AUDIENCE,
    AuthError,
    verify_mcp_token,
    verify_user_jwt,
)
from core.config import Settings, get_settings
from core.db import apply_app_migrations, set_tenant_context
from core.models import ChangePlan, RunContext
from evals.evaluators.judge import (
    TriageJudgeResult,
    compute_calibration_agreement,
    evaluate_triage_and_faithfulness,
)
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import InMemoryDomainStore
from telemetry.masking import PIISpanProcessor, _mask_attribute_value
from workflows.incident_desk.graph import (
    WorkflowDeps,
    _coerce_plan,
    build_incident_desk_graph,
)
from workflows.refund_ops.graph import build_refund_ops_graph


def test_auth_missing_and_malformed_claims_branches() -> None:
    secret = "auth-edge-secret-key-at-least-32-bytes-long"
    with pytest.raises(AuthError, match="Missing"):
        verify_user_jwt("", secret=secret)
    with pytest.raises(AuthError, match="Missing"):
        verify_mcp_token("", secret=secret)

    # Valid signature/audience but missing required claims
    bad_user = jwt.encode(
        {"aud": API_AUDIENCE, "tenant_id": "t1", "sub": "alice", "roles": "not-a-list"},
        secret,
        algorithm=JWT_ALGORITHM,
    )
    with pytest.raises(AuthError, match="Malformed user token"):
        verify_user_jwt(bad_user, secret=secret)

    bad_mcp = jwt.encode(
        {"aud": MCP_AUDIENCE, "tenant_id": "t1"},
        secret,
        algorithm=JWT_ALGORITHM,
    )
    with pytest.raises(AuthError, match="Malformed MCP token"):
        verify_mcp_token(bad_mcp, secret=secret)


def test_telemetry_masking_edge_branches() -> None:
    masked_list = _mask_attribute_value(["contact alice@corp.example", 42])
    assert masked_list[1] == 42 and "alice@corp.example" not in masked_list[0]
    masked_tuple = _mask_attribute_value(("sk-live-1234567890abcdef",))
    assert isinstance(masked_tuple, tuple) and "sk-live-" not in masked_tuple[0]

    proc = PIISpanProcessor()
    proc._scrub_span(SimpleNamespace(_attributes=None))  # type: ignore[arg-type]
    assert proc.shutdown() is None
    assert proc.force_flush() is True


async def test_judge_gateway_mode_and_blank_calibration_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FakeStructured:
        async def ainvoke(self, prompt: str) -> TriageJudgeResult:
            assert "rb-1" in prompt
            if "Read-Only Tool Observations:" in prompt:
                assert "get_service_status" in prompt
            return TriageJudgeResult(faithful=True, triage_score=5, rationale="ok")

    class _FakeModel:
        def with_structured_output(self, schema: Any) -> _FakeStructured:
            return _FakeStructured()

    monkeypatch.setattr("evals.evaluators.judge.get_chat_model", lambda **_: _FakeModel())

    gw_settings = Settings(model_mode="gateway")
    res = await evaluate_triage_and_faithfulness(
        summary="Roll back checkout-api",
        cited_doc_ids=["rb-1"],
        retrieved_runbooks=[{"doc_id": "rb-1", "title": "RB", "content": "Roll back"}],
        tool_observations=[
            'get_service_status: {"status": "degraded", "current_version": "v2.4.1"}'
        ],
        settings=gw_settings,
    )
    assert res.faithful is True and res.triage_score == 5

    cal_file = tmp_path / "cal.jsonl"
    cal_file.write_text(
        '\n{"summary":"Roll back","cited_doc_ids":["rb-1"],"retrieved_doc_ids":["rb-1"],'
        '"human_faithful":true,"human_triage_score":5}\n\n',
        encoding="utf-8",
    )
    cal = await compute_calibration_agreement(cal_file, settings=gw_settings)
    assert cal["samples"] == 1 and cal["agreement_rate"] == 1.0


async def test_api_auth_401_403_404_429_cancel_and_approval_edges() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )

    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        # 1. Missing / invalid Bearer -> 401
        assert (await client.get("/v1/threads")).status_code == 401
        assert (
            await client.get("/v1/threads", headers={"Authorization": "Bearer not-a-jwt"})
        ).status_code == 401

        alice_tok = (await client.post("/v1/dev/token", json={"persona": "alice"})).json()["token"]
        bob_tok = (await client.post("/v1/dev/token", json={"persona": "bob"})).json()["token"]
        alice_h = {"Authorization": f"Bearer {alice_tok}"}
        bob_h = {"Authorization": f"Bearer {bob_tok}"}

        # 2. 404s on unknown thread / run / stream / cancel / approval
        assert (
            await client.post(
                "/v1/threads/thr_missing/runs", headers=alice_h, json={"request_text": "hi"}
            )
        ).status_code == 404
        assert (await client.get("/v1/runs/run_missing", headers=alice_h)).status_code == 404
        assert (
            await client.get("/v1/runs/run_missing/stream", headers=alice_h)
        ).status_code == 404
        assert (
            await client.post("/v1/runs/run_missing/cancel", headers=alice_h)
        ).status_code == 404

        # 3. Role 403 on approvals list + 404 on unknown approval
        assert (await client.get("/v1/approvals", headers=alice_h)).status_code == 403
        assert (
            await client.post(
                "/v1/approvals/apr_missing/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
        ).status_code == 404

        # 4. Cancel queued follow-up while first run is STILL running -> keeps thread `running`
        t_id = (
            await client.post(
                "/v1/threads", headers=alice_h, json={"workflow_name": "incident_desk"}
            )
        ).json()["thread_id"]
        r1 = (
            await client.post(
                f"/v1/threads/{t_id}/runs", headers=alice_h, json={"request_text": "first"}
            )
        ).json()["run_id"]
        r2 = (
            await client.post(
                f"/v1/threads/{t_id}/runs",
                headers=alice_h,
                json={"request_text": "second", "multitask_strategy": "enqueue"},
            )
        ).json()["run_id"]
        c2 = await client.post(f"/v1/runs/{r2}/cancel", headers=alice_h)
        assert c2.status_code == 200
        assert c2.json()["thread_status"] == "running"

        # Cancel an already-cancelled run -> cancelled_before_claim=False
        c2_again = await client.post(f"/v1/runs/{r2}/cancel", headers=alice_h)
        assert c2_again.status_code == 200
        assert c2_again.json()["cancel_requested"] is False

        # Cancel r1 (no other active run) -> returns thread to `idle`
        c1 = await client.post(f"/v1/runs/{r1}/cancel", headers=alice_h)
        assert c1.status_code == 200 and c1.json()["thread_status"] == "idle"

        # 5. Per-tenant rate limit (>30 runs/min -> 429)
        t_rl = (
            await client.post(
                "/v1/threads", headers=alice_h, json={"workflow_name": "incident_desk"}
            )
        ).json()["thread_id"]
        got_429 = False
        for _ in range(35):
            r = await client.post(
                f"/v1/threads/{t_rl}/runs",
                headers=alice_h,
                json={"request_text": "rl", "multitask_strategy": "enqueue"},
            )
            if r.status_code == 429:
                got_429 = True
                break
        assert got_429, "Expected HTTP 429 after >30 runs/min for tenant-alpha"


async def test_workflow_helper_postgres_and_timeout_reconcile_succeeded_edges() -> None:
    # _coerce_plan None and ChangePlan instance
    assert _coerce_plan(None) is None
    sample_plan = ChangePlan(
        resource_type="service",
        target_resource="checkout-api",
        action="restart",
        risk_level="low",
        summary="restart",
    )
    assert _coerce_plan(sample_plan) is sample_plan

    # Missing RunContext raises ValueError in both workflows
    inc_graph = build_incident_desk_graph(checkpointer=InMemorySaver())
    with pytest.raises(ValueError, match="Missing required RunContext"):
        await inc_graph.ainvoke(
            {"request_text": "status only"},
            config={"configurable": {"thread_id": "no-ctx-inc"}},
        )
    ref_graph = build_refund_ops_graph(checkpointer=InMemorySaver())
    with pytest.raises(ValueError, match="Missing required RunContext"):
        await ref_graph.ainvoke(
            {"request_text": "balance only"},
            config={"configurable": {"thread_id": "no-ctx-ref"}},
        )

    # refund_ops with use_postgres=True in-process (hits retrieve_context & record_approval_request)
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

    t_pg = f"thr_ref_pg_{uuid.uuid4().hex[:8]}"
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
                    'refund_ops', '0.1.0+test', 1, 'running')
            """,
            (t_pg,),
        )
    pg_ref = build_refund_ops_graph(
        deps=WorkflowDeps(settings=settings, use_postgres=True, use_mcp_http=False),
        checkpointer=InMemorySaver(),
    )
    ctx_pg = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id=t_pg,
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        request_id="req-pg",
    )
    await pg_ref.ainvoke(
        {"request_text": "Issue $50 refund to ACC-1001"},
        config={"configurable": {"thread_id": t_pg}},
        context=ctx_pg,
    )
    snap_pg = await pg_ref.aget_state({"configurable": {"thread_id": t_pg}})
    assert snap_pg.interrupts and snap_pg.values["retrieved_runbooks"]

    # refund_ops get_account tool error path (missing ACC-9999) + timeout reconcile succeeded
    mem = InMemoryDomainStore()
    err_ref = build_refund_ops_graph(
        deps=WorkflowDeps(domain_store=mem, use_postgres=False, use_mcp_http=False),
        checkpointer=InMemorySaver(),
    )
    await err_ref.ainvoke(
        {"request_text": "balance only for ACC-9999"},
        config={"configurable": {"thread_id": "ref-acct-err"}},
        context=RunContext(
            tenant_id="tenant-alpha",
            requester_id="alice",
            requester_roles=frozenset({"engineer"}),
            thread_id="ref-acct-err",
            run_id="r-err",
            request_id="req-err",
        ),
    )

    # Timeout reconcile -> succeeded (operation already committed before timeout fired)
    to_ref = build_refund_ops_graph(
        deps=WorkflowDeps(
            domain_store=mem,
            use_postgres=False,
            use_mcp_http=False,
            simulate_write_timeout=True,
        ),
        checkpointer=InMemorySaver(),
    )
    cfg_rs = {"configurable": {"thread_id": "ref-to-ok"}}
    ctx_rs = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="ref-to-ok",
        run_id="r-rs",
        request_id="req-rs",
    )
    await to_ref.ainvoke(
        {"request_text": "Issue $25 refund to ACC-1001"}, config=cfg_rs, context=ctx_rs
    )
    intr_rs = (await to_ref.aget_state(cfg_rs)).interrupts[0].value
    # Pre-commit the operation so reconciliation discovers status == "succeeded"
    await mem.issue_refund(
        tenant_id="tenant-alpha",
        account_id="ACC-1001",
        amount_cents=2_500,
        reason="pre-committed",
        operation_id=intr_rs["operation_id"],
    )
    await to_ref.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": intr_rs["plan_hash"],
            }
        ),
        config=cfg_rs,
        context=ctx_rs,
    )
    assert (await to_ref.aget_state(cfg_rs)).values["final_status"] == "completed"

    # Same for incident_desk: timeout reconcile -> succeeded
    to_inc = build_incident_desk_graph(
        deps=WorkflowDeps(
            domain_store=mem,
            use_postgres=False,
            use_mcp_http=False,
            simulate_write_timeout=True,
        ),
        checkpointer=InMemorySaver(),
    )
    cfg_is = {"configurable": {"thread_id": "inc-to-ok"}}
    ctx_is = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="inc-to-ok",
        run_id="r-is",
        request_id="req-is",
    )
    await to_inc.ainvoke(
        {"request_text": "checkout-api returning 502s"}, config=cfg_is, context=ctx_is
    )
    intr_is = (await to_inc.aget_state(cfg_is)).interrupts[0].value
    await mem.execute_service_change(
        tenant_id="tenant-alpha",
        service_name="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        operation_id=intr_is["operation_id"],
    )
    await to_inc.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": intr_is["plan_hash"],
            }
        ),
        config=cfg_is,
        context=ctx_is,
    )
    assert (await to_inc.aget_state(cfg_is)).values["final_status"] == "completed"
