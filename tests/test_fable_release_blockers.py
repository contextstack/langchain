"""Boundary-first regression tests for all six release blockers found in Fable's audit.

1. Reused thread: a policy-denied run followed by an approved write on the same thread must
   clear stale per-run error state, run verification, and report `completed` / `verified=True`.
2. Redis outage: with Redis completely unreachable (`redis://127.0.0.1:1/0`), submitting,
   approving (HTTP 202, not 500), and worker execution must still succeed once via Postgres.
3. Approval Inbox XSS: `apps/web/index.html` must never interpolate plan/runbook data into
   `innerHTML`.
4. Failed-run trace redaction: OTel `PIISpanProcessor` must scrub span names & exception events,
   and `RedactingCallbackProxy` must forward & redact `on_llm_end` and all `*_error` callbacks.
5. Thread creation & approval API contracts: unknown `workflow_name` -> 422, authoritative
   `release_version` (never `0.1.0+dev`), invalid `edits` -> 422 (not 500), and 3 refused
   self-approval attempts -> 3 distinct `audit_log` rows.
6. Offline gateway embeddings: `check_embedding_ctx_length=False` so no tokenizer download occurs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
import psycopg
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from api.main import create_app
from core.config import Settings, get_settings
from core.db import apply_app_migrations, set_tenant_context
from core.models import RunContext
from gateway.factory import get_embeddings
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import InMemoryDomainStore
from telemetry.masking import PIISpanProcessor
from telemetry.tracing import RedactingCallbackProxy
from worker.main import RunWorker
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph


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


async def test_reused_thread_denied_then_approved_write_reports_completed_and_verified() -> None:
    store = InMemoryDomainStore()
    cp = InMemorySaver()
    graph = build_incident_desk_graph(
        deps=WorkflowDeps(domain_store=store, use_postgres=False, use_mcp_http=False),
        checkpointer=cp,
    )
    cfg = {"configurable": {"thread_id": "reused-thread-1"}}
    ctx1 = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="reused-thread-1",
        run_id="run-1-deny",
        request_id="req-1",
    )

    # Run 1: denied by PolicyEngine -> sets error_category="permission_denied", status="denied"
    await graph.ainvoke(
        {"request_text": "please drop_database on checkout-api"}, config=cfg, context=ctx1
    )
    snap1 = await graph.aget_state(cfg)
    assert snap1.values["final_status"] == "denied"
    assert snap1.values["error_category"] == "permission_denied"

    # Run 2 on the SAME thread: valid rollback -> approved -> MUST verify and report completed!
    ctx2 = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id="reused-thread-1",
        run_id="run-2-ok",
        request_id="req-2",
    )
    await graph.ainvoke(
        {"request_text": "checkout-api returning 502s after v2.4.1 deploy"},
        config=cfg,
        context=ctx2,
    )
    snap2 = await graph.aget_state(cfg)
    assert snap2.interrupts
    intr = snap2.interrupts[0].value

    await graph.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "approver_id": "bob",
                "approved_plan_hash": intr["plan_hash"],
            }
        ),
        config=cfg,
        context=ctx2,
    )
    end2 = await graph.aget_state(cfg)
    assert store.services[("tenant-alpha", "checkout-api")]["current_version"] == "v2.4.0"
    assert end2.values["error_category"] is None
    assert end2.values["verification_passed"] is True
    assert end2.values["final_status"] == "completed"


async def test_redis_outage_does_not_drop_approved_work() -> None:
    base = get_settings()
    await apply_app_migrations(base.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        base.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(base.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    # Point BOTH API and Worker at an unreachable Redis port
    offline_settings = base.model_copy(update={"redis_url": "redis://127.0.0.1:1/0"})
    app = create_app(offline_settings)
    worker = RunWorker(settings=offline_settings, worker_id="w-no-redis", poll_interval_s=0.1)

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
        r1_resp = await client.post(
            f"/v1/threads/{t_id}/runs",
            headers=alice_h,
            json={"request_text": "checkout-api returning 502s after v2.4.1 deploy"},
        )
        assert r1_resp.status_code == 202, r1_resp.text
        r1 = r1_resp.json()["run_id"]

        worker_task = asyncio.create_task(worker.serve_forever())
        try:
            await _poll_run(client, alice_h, r1, {"awaiting_approval"})
            approvals = (await client.get("/v1/approvals", headers=bob_h)).json()
            apr = next(a for a in approvals if a["thread_id"] == t_id)

            dec = await client.post(
                f"/v1/approvals/{apr['approval_id']}/decision",
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


def test_approval_inbox_html_never_uses_innerhtml_for_plan_fields() -> None:
    html = Path("apps/web/index.html").read_text(encoding="utf-8")
    assert "div.innerHTML" not in html
    assert "item.plan.summary" in html
    assert "textContent" in html


def test_failed_run_span_events_and_callback_errors_mask_all_pii() -> None:
    secret_email = "fail.user@corp.example"
    secret_key = "sk-live-9876543210abcdef1234"
    secret_phone = "415-555-0199"

    # 1. OTel PIISpanProcessor must scrub span name, attributes, AND recorded exception events
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(PIISpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("pii-fail-test")

    with tracer.start_as_current_span(
        f"run for {secret_email}",
        attributes={"input": f"key={secret_key}"},
    ) as span:
        span.record_exception(
            RuntimeError(f"boom for {secret_email} phone {secret_phone} key {secret_key}")
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    finished = spans[0]
    exported_text = json.dumps(
        {
            "name": finished.name,
            "attributes": dict(finished.attributes or {}),
            "events": [
                {"name": e.name, "attributes": dict(e.attributes or {})}
                for e in finished.events
            ],
        }
    )
    assert secret_email not in exported_text
    assert secret_key not in exported_text
    assert secret_phone not in exported_text

    # 2. RedactingCallbackProxy must forward & redact on_llm_end and all *_error callbacks
    seen: list[tuple[str, str]] = []

    class _CaptureHandler(BaseCallbackHandler):
        def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
            seen.append(("llm_end", str(response)))

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
            seen.append(("llm_error", str(error)))

        def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
            seen.append(("chain_error", str(error)))

        def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
            seen.append(("tool_error", str(error)))

    proxy = RedactingCallbackProxy(_CaptureHandler())
    llm_res = LLMResult(
        generations=[[Generation(text=f"Reply to {secret_email} with {secret_key}")]]
    )
    err = RuntimeError(f"Error for {secret_email} / {secret_phone} / {secret_key}")
    proxy.on_llm_end(llm_res)
    proxy.on_llm_error(err)
    proxy.on_chain_error(err)
    proxy.on_tool_error(err)

    assert [k for k, _ in seen] == ["llm_end", "llm_error", "chain_error", "tool_error"]
    all_seen = json.dumps(seen)
    assert secret_email not in all_seen
    assert secret_key not in all_seen
    assert secret_phone not in all_seen


async def test_thread_creation_validates_workflow_and_approval_edges() -> None:
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
    worker = RunWorker(settings=settings, worker_id="w-api-contracts", poll_interval_s=0.1)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        dana_tok = (await client.post("/v1/dev/token", json={"persona": "dana"})).json()["token"]
        bob_tok = (await client.post("/v1/dev/token", json={"persona": "bob"})).json()["token"]
        dana_h = {"Authorization": f"Bearer {dana_tok}"}
        bob_h = {"Authorization": f"Bearer {bob_tok}"}

        # 1. Unknown workflow_name -> 422 (NOT 201)
        bad_t = await client.post(
            "/v1/threads", headers=dana_h, json={"workflow_name": "does_not_exist"}
        )
        assert bad_t.status_code == 422, bad_t.text

        # 2. Valid thread gets authoritative release_version (NOT literal "0.1.0+dev")
        ok_t = await client.post(
            "/v1/threads",
            headers=dana_h,
            json={"workflow_name": "incident_desk", "release_version": "0.1.0+dev"},
        )
        assert ok_t.status_code == 201
        t_body = ok_t.json()
        assert t_body["release_version"] != "0.1.0+dev"
        t_id = t_body["thread_id"]

        r1 = (
            await client.post(
                f"/v1/threads/{t_id}/runs",
                headers=dana_h,
                json={"request_text": "checkout-api returning 502s after v2.4.1 deploy"},
            )
        ).json()["run_id"]

        worker_task = asyncio.create_task(worker.serve_forever())
        try:
            await _poll_run(client, dana_h, r1, {"awaiting_approval"})
            approvals = (await client.get("/v1/approvals", headers=bob_h)).json()
            apr_id = next(a["approval_id"] for a in approvals if a["thread_id"] == t_id)

            # 3. Invalid edit payload -> HTTP 422 (NOT HTTP 500)
            bad_edit = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                headers=bob_h,
                json={"decision": "edit", "edits": {"risk_level": "not-a-valid-risk"}},
            )
            assert bad_edit.status_code == 422, bad_edit.text

            # 4. Three refused self-approval attempts by Dana -> 3 distinct audit_log rows!
            for _ in range(3):
                refused = await client.post(
                    f"/v1/approvals/{apr_id}/decision",
                    headers=dana_h,
                    json={"decision": "approve"},
                )
                assert refused.status_code == 403

            async with (
                await psycopg.AsyncConnection.connect(settings.database_url) as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                await set_tenant_context(conn, "tenant-alpha")
                await cur.execute(
                    """
                    SELECT count(*)
                      FROM audit_log
                     WHERE thread_id = %s
                       AND event_type = 'approval_refused'
                       AND decision = 'refused'
                    """,
                    (t_id,),
                )
                assert (await cur.fetchone())[0] == 3
        finally:
            worker.request_stop()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker_task, timeout=5.0)


def test_gateway_embeddings_disables_tiktoken_download() -> None:
    emb = get_embeddings(tenant_id="tenant-alpha", settings=Settings(model_mode="gateway"))
    assert getattr(emb, "check_embedding_ctx_length", True) is False
