"""M3-M5 Integration Tests & Real-Process Failure Drills (AC-1..AC-16)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psycopg

from api.main import create_app
from core.auth import mint_user_jwt
from core.config import get_settings
from core.db import apply_app_migrations
from core.models import CallerIdentity
from evals.runner import run_eval_suite
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain


async def _reset_db() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            """
            TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;
            """
        )
    async with await psycopg.AsyncConnection.connect(
        settings.database_url, autocommit=True
    ) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())


def _token_for(actor: str, tenant: str, roles: set[str]) -> dict[str, str]:
    settings = get_settings()
    tok = mint_user_jwt(
        CallerIdentity(tenant_id=tenant, actor_id=actor, roles=frozenset(roles)),
        secret=settings.jwt_signing_secret,
    )
    return {"Authorization": f"Bearer {tok}"}


def _spawn_worker(extra_env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.setdefault("WORKER_LEASE_S", "2")
    env.setdefault("WORKER_HEARTBEAT_S", "1")
    env.setdefault("MODEL_MODE", "mock")
    env.setdefault("TELEMETRY_BACKEND", "none")
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-c", "from worker.main import run; run()"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_proc(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _wait_for_run_status(
    client: httpx.AsyncClient,
    run_id: str,
    headers: dict[str, str],
    targets: set[str],
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resp = await client.get(f"/v1/runs/{run_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in targets:
            return last
        await asyncio.sleep(0.2)
    raise AssertionError(f"Run {run_id} did not reach {targets} within {timeout_s}s; last={last}")


async def _count_mock_ops() -> int:
    settings = get_settings()
    async with (
        await psycopg.AsyncConnection.connect(settings.database_url) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute("SELECT COUNT(*) FROM mock_operations")
        row = await cur.fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# AC-1: Worker killed (os._exit(137)) mid-inference -> automatic reclaim
# ---------------------------------------------------------------------------
async def test_drill_ac1_crash_mid_inference() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})

    w_flaky = _spawn_worker({"FAULT_INJECT": "investigate", "WORKER_LEASE_S": "2"})
    w_healthy: subprocess.Popen[bytes] | None = None
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api is returning 502s"},
                    headers=alice,
                )
            ).json()["run_id"]

            # Wait for flaky worker to crash via os._exit(137)
            w_flaky.wait(timeout=10)
            assert w_flaky.returncode == 137

            # Start healthy worker -> reclaims automatically after 2s lease expiry
            w_healthy = _spawn_worker({"WORKER_LEASE_S": "2"})
            final_run = await _wait_for_run_status(
                client, r_id, alice, {"awaiting_approval"}, timeout_s=15.0
            )
            assert final_run["attempt"] >= 2
            assert final_run["thread_status"] == "awaiting_approval"

            # Verify run_reclaimed audit row exists
            audits = (await client.get(f"/v1/audit?thread_id={t_id}", headers=alice)).json()
            assert any(a["event_type"] == "run_reclaimed" for a in audits)
    finally:
        _stop_proc(w_flaky)
        if w_healthy:
            _stop_proc(w_healthy)


# ---------------------------------------------------------------------------
# AC-2: Worker killed AFTER external write commit -> exactly-once mutation
# ---------------------------------------------------------------------------
async def test_drill_ac2_crash_after_write() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})

    w1 = _spawn_worker({"WORKER_LEASE_S": "2"})
    w_crash: subprocess.Popen[bytes] | None = None
    w2: subprocess.Popen[bytes] | None = None
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api is returning 502s"},
                    headers=alice,
                )
            ).json()["run_id"]
            await _wait_for_run_status(client, r_id, alice, {"awaiting_approval"})
            _stop_proc(w1)

            approvals = (await client.get("/v1/approvals", headers=bob)).json()
            assert len(approvals) == 1
            apr_id = approvals[0]["approval_id"]

            # Start worker that crashes RIGHT AFTER mock_operations commits
            w_crash = _spawn_worker({"FAULT_INJECT": "after_write", "WORKER_LEASE_S": "2"})
            dec = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve", "comment": "go"},
                headers=bob,
            )
            assert dec.status_code == 202
            resume_run_id = dec.json()["run_id"]

            w_crash.wait(timeout=10)
            assert w_crash.returncode == 137

            # Healthy worker reclaims -> replays with same operation_id -> deduplicated!
            w2 = _spawn_worker({"WORKER_LEASE_S": "2"})
            res = await _wait_for_run_status(
                client, resume_run_id, alice, {"succeeded"}, timeout_s=15.0
            )
            assert res["attempt"] >= 2
            assert await _count_mock_ops() == 1
    finally:
        _stop_proc(w1)
        if w_crash:
            _stop_proc(w_crash)
        if w2:
            _stop_proc(w2)


# ---------------------------------------------------------------------------
# AC-3: Full API + worker restart while paused at interrupt()
# ---------------------------------------------------------------------------
async def test_drill_ac3_restart_hitl() -> None:
    await _reset_db()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})

    w1 = _spawn_worker({"WORKER_LEASE_S": "5"})
    app1 = create_app()
    apr_id = ""
    try:
        async with (
            app1.router.lifespan_context(app1),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app1), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice,
                )
            ).json()["run_id"]
            await _wait_for_run_status(client, r_id, alice, {"awaiting_approval"})
            apr_id = (await client.get("/v1/approvals", headers=bob)).json()[0]["approval_id"]
    finally:
        _stop_proc(w1)

    # Completely fresh API app + fresh worker process
    w2 = _spawn_worker({"WORKER_LEASE_S": "5"})
    app2 = create_app()
    try:
        async with (
            app2.router.lifespan_context(app2),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app2), base_url="http://test"
            ) as client,
        ):
            dec = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve", "comment": "post-restart"},
                headers=bob,
            )
            assert dec.status_code == 202
            res = await _wait_for_run_status(client, dec.json()["run_id"], alice, {"succeeded"})
            assert res["thread_status"] == "completed"
            assert await _count_mock_ops() == 1
    finally:
        _stop_proc(w2)


# ---------------------------------------------------------------------------
# AC-4: Cross-tenant isolation on every API endpoint
# ---------------------------------------------------------------------------
async def test_drill_ac4_tenant_isolation() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})
    carol_beta = _token_for("carol", "tenant-beta", {"engineer", "ops-approver"})

    w = _spawn_worker({"WORKER_LEASE_S": "5"})
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice,
                )
            ).json()["run_id"]
            await _wait_for_run_status(client, r_id, alice, {"awaiting_approval"})
            apr_id = (await client.get("/v1/approvals", headers=bob)).json()[0]["approval_id"]

            # tenant-beta gets 404 on every tenant-alpha ID
            assert (await client.get(f"/v1/runs/{r_id}", headers=carol_beta)).status_code == 404
            assert (
                await client.post(f"/v1/runs/{r_id}/cancel", headers=carol_beta)
            ).status_code == 404
            assert (
                await client.post(
                    f"/v1/approvals/{apr_id}/decision",
                    json={"decision": "approve"},
                    headers=carol_beta,
                )
            ).status_code == 404
            assert (await client.get("/v1/threads", headers=carol_beta)).json() == []
            assert (await client.get("/v1/approvals", headers=carol_beta)).json() == []
            assert (await client.get("/v1/audit", headers=carol_beta)).json() == []
    finally:
        _stop_proc(w)


# ---------------------------------------------------------------------------
# AC-6: Incompatible checkpoint_compat fails with migration_required
# ---------------------------------------------------------------------------
async def test_drill_ac6_version_upgrade() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})

    w_v1 = _spawn_worker({"CHECKPOINT_COMPAT_OVERRIDE": "1"})
    w_v2: subprocess.Popen[bytes] | None = None
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (
                await client.post(
                    "/v1/threads", json={"checkpoint_compat": 1}, headers=alice
                )
            ).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice,
                )
            ).json()["run_id"]
            await _wait_for_run_status(client, r_id, alice, {"awaiting_approval"})
            apr_id = (await client.get("/v1/approvals", headers=bob)).json()[0]["approval_id"]
            _stop_proc(w_v1)

            # Deploy v2 worker with incompatible checkpoint_compat=2
            w_v2 = _spawn_worker({"CHECKPOINT_COMPAT_OVERRIDE": "2"})
            dec = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve"},
                headers=bob,
            )
            assert dec.status_code == 202
            res = await _wait_for_run_status(client, dec.json()["run_id"], alice, {"failed"})
            assert res["error_detail"] == "migration_required"
            assert res["thread_status"] == "blocked_version"
            assert await _count_mock_ops() == 0

            audits = (await client.get(f"/v1/audit?thread_id={t_id}", headers=alice)).json()
            assert any(a["event_type"] == "version_blocked" for a in audits)
    finally:
        _stop_proc(w_v1)
        if w_v2:
            _stop_proc(w_v2)


# ---------------------------------------------------------------------------
# AC-9: Separation of duties (self-approval refused & audited; bob can approve)
# ---------------------------------------------------------------------------
async def test_drill_ac9_self_approval() -> None:
    await _reset_db()
    app = create_app()
    dana = _token_for("dana", "tenant-alpha", {"engineer", "ops-approver"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})

    w = _spawn_worker()
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=dana)).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=dana,
                )
            ).json()["run_id"]
            await _wait_for_run_status(client, r_id, dana, {"awaiting_approval"})
            apr_id = (await client.get("/v1/approvals", headers=dana)).json()[0]["approval_id"]

            # Dana tries to approve her own request -> 403 + approval_refused audit
            bad = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve"},
                headers=dana,
            )
            assert bad.status_code == 403
            audits = (await client.get(f"/v1/audit?thread_id={t_id}", headers=dana)).json()
            assert any(a["event_type"] == "approval_refused" for a in audits)

            # Bob can still approve and complete it
            good = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve"},
                headers=bob,
            )
            assert good.status_code == 202
            res = await _wait_for_run_status(client, good.json()["run_id"], dana, {"succeeded"})
            assert res["thread_status"] == "completed"
    finally:
        _stop_proc(w)


# ---------------------------------------------------------------------------
# AC-10: Double-approve concurrency + single active run per thread
# ---------------------------------------------------------------------------
async def test_drill_ac10_double_approve_and_single_run() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})
    dana = _token_for("dana", "tenant-alpha", {"engineer", "ops-approver"})

    w = _spawn_worker()
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r1 = await client.post(
                f"/v1/threads/{t_id}/runs",
                json={"request_text": "checkout-api returning 502s"},
                headers=alice,
            )
            assert r1.status_code == 202

            # Second start on busy thread -> 409
            r2 = await client.post(
                f"/v1/threads/{t_id}/runs",
                json={"request_text": "duplicate start"},
                headers=alice,
            )
            assert r2.status_code == 409

            await _wait_for_run_status(client, r1.json()["run_id"], alice, {"awaiting_approval"})
            apr_id = (await client.get("/v1/approvals", headers=bob)).json()[0]["approval_id"]

            # Two approvers decide concurrently -> exactly one 202 and one 409
            resp_a, resp_b = await asyncio.gather(
                client.post(
                    f"/v1/approvals/{apr_id}/decision",
                    json={"decision": "approve"},
                    headers=bob,
                ),
                client.post(
                    f"/v1/approvals/{apr_id}/decision",
                    json={"decision": "approve"},
                    headers=dana,
                ),
            )
            codes = sorted([resp_a.status_code, resp_b.status_code])
            assert codes == [202, 409]

            winner = resp_a if resp_a.status_code == 202 else resp_b
            await _wait_for_run_status(client, winner.json()["run_id"], alice, {"succeeded"})
            assert await _count_mock_ops() == 1
    finally:
        _stop_proc(w)


# ---------------------------------------------------------------------------
# AC-14: Cancel before claim (queued run cancelled atomically, thread idle)
# ---------------------------------------------------------------------------
async def test_ac14_cancel_before_claim() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})

    # No worker running -> run stays queued
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
        r1 = await client.post(
            f"/v1/threads/{t_id}/runs",
            json={"request_text": "checkout-api returning 502s"},
            headers=alice,
        )
        run_id = r1.json()["run_id"]

        cancel_resp = await client.post(f"/v1/runs/{run_id}/cancel", headers=alice)
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancelled_before_claim"] is True

        run_info = (await client.get(f"/v1/runs/{run_id}", headers=alice)).json()
        assert run_info["status"] == "cancelled"
        assert run_info["attempt"] == 0
        assert run_info["thread_status"] == "idle"

        # Freed thread accepts a fresh start run
        r2 = await client.post(
            f"/v1/threads/{t_id}/runs",
            json={"request_text": "fresh start"},
            headers=alice,
        )
        assert r2.status_code == 202


# ---------------------------------------------------------------------------
# AC-16: Cancel after downstream commit -> reconciles instead of hiding commit
# ---------------------------------------------------------------------------
async def test_drill_ac16_cancel_after_write_reconciles() -> None:
    await _reset_db()
    app = create_app()
    alice = _token_for("alice", "tenant-alpha", {"engineer"})
    bob = _token_for("bob", "tenant-alpha", {"ops-approver"})

    w1 = _spawn_worker({"WORKER_LEASE_S": "2"})
    w_crash: subprocess.Popen[bytes] | None = None
    w2: subprocess.Popen[bytes] | None = None
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            t_id = (await client.post("/v1/threads", json={}, headers=alice)).json()["thread_id"]
            r_id = (
                await client.post(
                    f"/v1/threads/{t_id}/runs",
                    json={"request_text": "checkout-api returning 502s"},
                    headers=alice,
                )
            ).json()["run_id"]
            await _wait_for_run_status(client, r_id, alice, {"awaiting_approval"})
            apr_id = (await client.get("/v1/approvals", headers=bob)).json()[0]["approval_id"]
            _stop_proc(w1)

            # Worker commits write to mock_operations then exits via FAULT_INJECT=after_write
            w_crash = _spawn_worker({"FAULT_INJECT": "after_write", "WORKER_LEASE_S": "2"})
            dec = await client.post(
                f"/v1/approvals/{apr_id}/decision",
                json={"decision": "approve"},
                headers=bob,
            )
            resume_run_id = dec.json()["run_id"]
            w_crash.wait(timeout=10)
            assert w_crash.returncode == 137

            # Caller cancels while run is still marked `running` (lease expiring)
            cancel_resp = await client.post(f"/v1/runs/{resume_run_id}/cancel", headers=alice)
            assert cancel_resp.status_code == 200
            assert cancel_resp.json()["cancel_requested"] is True

            # Sweeping worker reconciles via get_operation_status -> reports confirmed commit!
            w2 = _spawn_worker({"WORKER_LEASE_S": "2"})
            res = await _wait_for_run_status(
                client, resume_run_id, alice, {"succeeded", "failed", "cancelled"}, timeout_s=15.0
            )
            assert res["status"] == "succeeded"
            assert res["thread_status"] == "completed"
            assert await _count_mock_ops() == 1

            audits = (await client.get(f"/v1/audit?thread_id={t_id}", headers=alice)).json()
            assert any(
                a["event_type"] == "reconciliation" and a["decision"] == "succeeded"
                for a in audits
            )
    finally:
        _stop_proc(w1)
        if w_crash:
            _stop_proc(w_crash)
        if w2:
            _stop_proc(w2)


# ---------------------------------------------------------------------------
# AC-7: Offline evaluation regression gate + calibration agreement
# ---------------------------------------------------------------------------
async def test_ac7_offline_eval_regression_gate() -> None:
    report = await run_eval_suite(
        Path("evals/datasets/core.jsonl"),
        Path("evals/datasets/calibration.jsonl"),
    )
    assert report["passed"] is True
    assert report["total"] == 6
    assert report["calibration"] is not None
    assert report["calibration"]["agreement_rate"] == 1.0
