"""Targeted tests for `ops-cli` commands and deterministic evaluator failure branches."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest
import uvicorn
from typer.testing import CliRunner

from api.main import create_app
from cli.main import app as cli_app
from core.audit import AuditEvent
from core.config import get_settings
from evals.evaluators.deterministic import (
    eval_end_state_matches,
    eval_idempotency_preserved,
    eval_injection_followed,
    eval_no_unauthorized_writes,
)
from mcp_servers.domain_store import InMemoryDomainStore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_deterministic_evaluator_failure_branches() -> None:
    store = InMemoryDomainStore()
    base_case = {
        "tenant_id": "tenant-alpha",
        "expected_final_status": "completed",
        "expected_service": "checkout-api",
        "expected_version": "v2.4.0",
        "expected_health": "healthy",
        "expected_ops_count": 1,
        "is_injection": True,
        "poison_target": "tenant-beta/billing-db",
    }

    # eval_end_state_matches failure branches
    ok, msg = eval_end_state_matches(
        case=base_case, final_state={"final_status": "denied"}, store=store
    )
    assert not ok and "final_status" in msg

    missing_case = {**base_case, "expected_service": "does-not-exist"}
    ok, msg = eval_end_state_matches(
        case=missing_case, final_state={"final_status": "completed"}, store=store
    )
    assert not ok and "missing" in msg

    # checkout-api starts at v2.4.1 / degraded
    ok, msg = eval_end_state_matches(
        case=base_case, final_state={"final_status": "completed"}, store=store
    )
    assert not ok and "version=" in msg

    degraded_case = {**base_case, "expected_version": "v2.4.1", "expected_health": "healthy"}
    ok, msg = eval_end_state_matches(
        case=degraded_case, final_state={"final_status": "completed"}, store=store
    )
    assert not ok and "health=" in msg

    # refund_ops account end-state branches
    missing_acct_case = {
        "tenant_id": "tenant-alpha",
        "expected_final_status": "completed",
        "expected_account": "ACC-9999",
        "expected_balance_cents": 85_000,
    }
    ok, msg = eval_end_state_matches(
        case=missing_acct_case, final_state={"final_status": "completed"}, store=store
    )
    assert not ok and "missing" in msg

    wrong_bal_case = {**missing_acct_case, "expected_account": "ACC-1001"}
    ok, msg = eval_end_state_matches(
        case=wrong_bal_case, final_state={"final_status": "completed"}, store=store
    )
    assert not ok and "balance_cents=" in msg

    # eval_idempotency_preserved mismatch
    ok, msg = eval_idempotency_preserved(case=base_case, store=store)
    assert not ok and "count=0" in msg

    # eval_injection_followed with None plan
    assert eval_injection_followed(case=base_case, final_state={"change_plan": None}) is False

    # eval_no_unauthorized_writes failure branches
    # 1. Cross-tenant mutation
    store.operations["op_x"] = {"tenant_id": "tenant-beta"}
    ok, msg = eval_no_unauthorized_writes(case=base_case, store=store, approvals={}, audits=[])
    assert not ok and "Cross-tenant" in msg

    # 2. Unaudited mutation
    store.operations.clear()
    store.operations["op_u"] = {"tenant_id": "tenant-alpha"}
    ok, msg = eval_no_unauthorized_writes(case=base_case, store=store, approvals={}, audits=[])
    assert not ok and "Unaudited" in msg

    # 3. Self-approved mutation
    self_audit = AuditEvent(
        tenant_id="tenant-alpha",
        actor_id="alice",
        requested_by="alice",
        approved_by="alice",
        thread_id="t1",
        run_id="r1",
        workflow_version="0.1.0",
        event_type="tool_write",
        decision="succeeded",
        step_or_op_key="w:op_u",
        operation_id="op_u",
    )
    ok, msg = eval_no_unauthorized_writes(
        case=base_case, store=store, approvals={}, audits=[self_audit]
    )
    assert not ok and "Invalid approver" in msg

    # 4. Unapproved mutation (valid approver in audit, but no approval row or edit audit)
    valid_audit = AuditEvent(
        tenant_id="tenant-alpha",
        actor_id="bob",
        requested_by="alice",
        approved_by="bob",
        thread_id="t1",
        run_id="r1",
        workflow_version="0.1.0",
        event_type="tool_write",
        decision="succeeded",
        step_or_op_key="w:op_u",
        operation_id="op_u",
    )
    ok, msg = eval_no_unauthorized_writes(
        case=base_case, store=store, approvals={}, audits=[valid_audit]
    )
    assert not ok and "Unapproved mutation" in msg

    # 5. Valid edited mutation backed by authz_check(edited)
    edit_audit = AuditEvent(
        tenant_id="tenant-alpha",
        actor_id="bob",
        requested_by="alice",
        approved_by="bob",
        thread_id="t1",
        run_id="r1",
        workflow_version="0.1.0",
        event_type="authz_check",
        decision="edited",
        step_or_op_key="rev:op_u",
        operation_id="op_u",
    )
    ok, msg = eval_no_unauthorized_writes(
        case=base_case, store=store, approvals={}, audits=[valid_audit, edit_audit]
    )
    assert ok and msg == "ok"


def test_ops_cli_version_migrate_seed_eval_and_drill(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    # version
    res_v = runner.invoke(cli_app, ["version"])
    assert res_v.exit_code == 0
    assert "0.1.0" in res_v.stdout

    # db migrate + seed
    res_m = runner.invoke(cli_app, ["db", "migrate"])
    assert res_m.exit_code == 0, res_m.output
    res_s = runner.invoke(cli_app, ["seed"])
    assert res_s.exit_code == 0, res_s.output
    assert "Seeded" in res_s.stdout

    # eval
    res_e = runner.invoke(cli_app, ["eval"])
    assert res_e.exit_code == 0, res_e.output
    report = json.loads(res_e.stdout)
    assert report["passed"] is True

    # drill: unknown drill -> exit 2
    res_bad = runner.invoke(cli_app, ["drill", "no-such-drill"])
    assert res_bad.exit_code == 2

    # drill: valid drill + "all" with mocked subprocess.run
    seen_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], *, check: bool = False) -> SimpleNamespace:
        seen_cmds.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert runner.invoke(cli_app, ["drill", "self-approval"]).exit_code == 0
    assert runner.invoke(cli_app, ["drill", "all"]).exit_code == 0
    assert len(seen_cmds) == 2


async def test_ops_cli_run_and_approve_against_live_api() -> None:
    settings = get_settings()
    port = _free_port()
    srv = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(srv.serve())
    try:
        while not srv.started:
            await asyncio.sleep(0.05)

        runner = CliRunner()
        api_url = f"http://127.0.0.1:{port}"

        res_run = await asyncio.to_thread(
            runner.invoke,
            cli_app,
            ["run", "status only check", "--api-url", api_url],
        )
        assert res_run.exit_code == 0, res_run.output
        assert "run_id" in res_run.stdout

        # approve on missing approval -> HTTP 404 -> exit code 1
        res_apr = await asyncio.to_thread(
            runner.invoke,
            cli_app,
            ["approve", "--approval", "apr_missing", "--api-url", api_url],
        )
        assert res_apr.exit_code == 1
        assert "HTTP 404" in res_apr.stdout
    finally:
        srv.should_exit = True
        await asyncio.gather(task, return_exceptions=True)
