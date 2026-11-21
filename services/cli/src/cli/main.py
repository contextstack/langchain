"""Typer CLI entrypoint (`ops-cli`): migrate, seed, run, approve, eval, drill."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import psycopg
import typer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from core.auth import mint_user_jwt
from core.config import get_settings
from core.db import apply_app_migrations
from core.models import CallerIdentity
from evals.runner import run_eval_suite
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain

app = typer.Typer(name="ops-cli", help="ContextStack Enterprise LangChain CLI")
db_app = typer.Typer(name="db", help="Database migration and maintenance commands")
app.add_typer(db_app, name="db")

DRILL_TO_PYTEST: dict[str, str] = {
    "crash-mid-inference": (
        "tests/test_m3_m5_api_worker_drills.py::test_drill_ac1_crash_mid_inference"
    ),
    "crash-after-write": "tests/test_m3_m5_api_worker_drills.py::test_drill_ac2_crash_after_write",
    "restart-hitl": "tests/test_m3_m5_api_worker_drills.py::test_drill_ac3_restart_hitl",
    "tenant-isolation": "tests/test_m3_m5_api_worker_drills.py::test_drill_ac4_tenant_isolation",
    "prompt-injection-cross-tenant": (
        "tests/test_m2_workflow_and_telemetry.py::"
        "test_m2_ac5_prompt_injection_cross_tenant_and_in_tenant"
    ),
    "prompt-injection-in-tenant": (
        "tests/test_m2_workflow_and_telemetry.py::"
        "test_m2_ac5_prompt_injection_cross_tenant_and_in_tenant"
    ),
    "version-upgrade": "tests/test_m3_m5_api_worker_drills.py::test_drill_ac6_version_upgrade",
    "self-approval": "tests/test_m3_m5_api_worker_drills.py::test_drill_ac9_self_approval",
    "double-approve": (
        "tests/test_m3_m5_api_worker_drills.py::test_drill_ac10_double_approve_and_single_run"
    ),
    "edit-recheck": (
        "tests/test_m2_workflow_and_telemetry.py::"
        "test_m2_ac13_edit_rechecked_by_policy_and_new_operation_id"
    ),
    "cancel-after-write": (
        "tests/test_m3_m5_api_worker_drills.py::test_drill_ac16_cancel_after_write_reconciles"
    ),
}


@app.callback()
def main() -> None:
    """ContextStack Enterprise LangChain CLI."""


@app.command()
def version() -> None:
    """Print the CLI version."""
    typer.echo("contextstack-langchain 0.1.0")


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply application schemas, DB role grants, append-only audit trigger, and LangGraph setup."""

    async def _migrate() -> None:
        settings = get_settings()
        await apply_app_migrations(settings.database_admin_url)
        async with (
            AsyncPostgresSaver.from_conn_string(settings.database_admin_url) as checkpointer,
            AsyncPostgresStore.from_conn_string(settings.database_admin_url) as store,
        ):
            await checkpointer.setup()
            await store.setup()

        async with await psycopg.AsyncConnection.connect(
            settings.database_admin_url, autocommit=True
        ) as conn:
            await conn.execute(
                """
                GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_rw;
                REVOKE ALL ON audit_log FROM app_rw;
                GRANT SELECT, INSERT ON audit_log TO app_rw;
                """
            )

    asyncio.run(_migrate())
    typer.echo("Migrations and LangGraph checkpointer/store setup complete.")


@app.command()
def seed() -> None:
    """Idempotently seed multi-tenant runbooks (incl. injection fixtures), services, and tickets."""

    async def _seed() -> dict[str, int]:
        settings = get_settings()
        embeddings = DeterministicMockEmbeddings()
        async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
            return await seed_knowledge_and_domain(conn, embeddings=embeddings)

    counts = asyncio.run(_seed())
    typer.echo(
        f"Seeded {counts['runbooks']} runbooks, "
        f"{counts['services']} services, "
        f"{counts['tickets']} tickets."
    )


@app.command()
def run(
    request_text: str = typer.Argument(..., help="Incident or change request text"),
    tenant: str = typer.Option("tenant-alpha", help="Tenant ID"),
    actor: str = typer.Option("alice", help="Actor ID"),
    role: str = typer.Option("engineer", help="Caller role"),
    api_url: str = typer.Option("http://127.0.0.1:8000", help="FastAPI base URL"),
) -> None:
    """Create a thread and enqueue a start run via the FastAPI serving plane."""
    settings = get_settings()
    token = mint_user_jwt(
        CallerIdentity(tenant_id=tenant, actor_id=actor, roles=frozenset({role})),
        secret=settings.jwt_signing_secret,
    )
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=api_url, headers=headers, timeout=15.0) as client:
        t = client.post("/v1/threads", json={"workflow_name": "incident_desk"})
        t.raise_for_status()
        thread_id = t.json()["thread_id"]
        r = client.post(f"/v1/threads/{thread_id}/runs", json={"request_text": request_text})
        r.raise_for_status()
        typer.echo(json.dumps(r.json(), indent=2))


@app.command()
def approve(
    approval: str = typer.Option(..., "--approval", help="Approval ID"),
    decision: str = typer.Option("approve", help="approve | edit | reject"),
    tenant: str = typer.Option("tenant-alpha", help="Approver tenant ID"),
    actor: str = typer.Option("bob", help="Approver actor ID"),
    comment: str = typer.Option("Approved via ops-cli", help="Decision comment"),
    api_url: str = typer.Option("http://127.0.0.1:8000", help="FastAPI base URL"),
) -> None:
    """Submit an approval decision via the FastAPI serving plane."""
    settings = get_settings()
    token = mint_user_jwt(
        CallerIdentity(tenant_id=tenant, actor_id=actor, roles=frozenset({"ops-approver"})),
        secret=settings.jwt_signing_secret,
    )
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=api_url, headers=headers, timeout=15.0) as client:
        resp = client.post(
            f"/v1/approvals/{approval}/decision",
            json={"decision": decision, "comment": comment},
        )
        typer.echo(f"HTTP {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise typer.Exit(code=1)


@app.command("eval")
def run_eval(
    dataset: Path = typer.Option(  # noqa: B008
        Path("evals/datasets/core.jsonl"),
        "--dataset",
        help="Path to JSONL evaluation dataset",
    ),
    calibration: Path = typer.Option(  # noqa: B008
        Path("evals/datasets/calibration.jsonl"),
        "--calibration",
        help="Path to hand-labeled calibration JSONL",
    ),
) -> None:
    """Run the offline evaluation suite and exit non-zero on regression (AC-7)."""
    report = asyncio.run(run_eval_suite(dataset, calibration))
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command()
def drill(
    name: str = typer.Argument(
        ...,
        help=f"Drill name ({', '.join(sorted(DRILL_TO_PYTEST))} or 'all')",
    ),
) -> None:
    """Run one or all enterprise failure/security drills."""
    targets = (
        ["tests/test_m3_m5_api_worker_drills.py", "tests/test_m2_workflow_and_telemetry.py"]
        if name == "all"
        else [DRILL_TO_PYTEST[name]]
        if name in DRILL_TO_PYTEST
        else []
    )
    if not targets:
        typer.echo(f"Unknown drill '{name}'. Choices: {', '.join(sorted(DRILL_TO_PYTEST))}, all")
        raise typer.Exit(code=2)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-v", *targets], check=False)
    raise typer.Exit(code=proc.returncode)
