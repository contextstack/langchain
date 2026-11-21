"""FastAPI Serving Plane (spec §8.1).

Strictly adheres to workspace boundary: NEVER imports `packages/workflows`.
Authenticates callers, writes `app_threads`/`runs`/`approvals` rows, and streams
SSE events from Redis Streams.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from core.audit import AuditEvent, write_audit_event
from core.auth import AuthError, mint_user_jwt, verify_user_jwt
from core.config import Settings, get_settings
from core.db import set_tenant_context
from core.models import CallerIdentity, ChangePlan
from core.policy import PolicyDecision, PolicyEngine

WAKEUP_CHANNEL = "ops:runs:wakeup"
CANCEL_CHANNEL = "ops:runs:cancel"

# Dev persona catalog for the local single-file Demo Web UI
DEV_PERSONAS: dict[str, CallerIdentity] = {
    "alice": CallerIdentity(
        tenant_id="tenant-alpha",
        actor_id="alice",
        roles=frozenset({"engineer"}),
    ),
    "bob": CallerIdentity(
        tenant_id="tenant-alpha",
        actor_id="bob",
        roles=frozenset({"ops-approver"}),
    ),
    "carol": CallerIdentity(
        tenant_id="tenant-beta",
        actor_id="carol",
        roles=frozenset({"engineer"}),
    ),
    "dana": CallerIdentity(
        tenant_id="tenant-alpha",
        actor_id="dana",
        roles=frozenset({"engineer", "ops-approver"}),
    ),
}


class CreateThreadRequest(BaseModel):
    workflow_name: str = "incident_desk"
    release_version: str = "0.1.0+dev"
    checkpoint_compat: int = 1


class StartRunRequest(BaseModel):
    request_text: str = Field(min_length=1)
    multitask_strategy: Literal["reject", "enqueue", "interrupt"] = "reject"


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    edits: dict[str, Any] | None = None
    comment: str = ""


class DevTokenRequest(BaseModel):
    persona: Literal["alice", "bob", "carol", "dana"]


def _web_index_path() -> Path:
    return Path(__file__).resolve().parents[4] / "apps" / "web" / "index.html"


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    policy = PolicyEngine()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            conninfo=cfg.database_url,
            min_size=2,
            max_size=10,
            open=False,
        )
        await pool.open()
        rds = aioredis.from_url(cfg.redis_url, decode_responses=True)
        app.state.pool = pool
        app.state.redis = rds
        app.state.settings = cfg
        try:
            yield
        finally:
            await rds.aclose()
            await pool.close()

    app = FastAPI(
        title="ContextStack Enterprise LangChain API",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def get_caller(
        authorization: str | None = Header(default=None),
    ) -> CallerIdentity:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Bearer token",
            )
        token = authorization[7:].strip()
        try:
            return verify_user_jwt(token, secret=cfg.jwt_signing_secret)
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
            ) from exc

    async def _enforce_rate_limit(request: Request, tenant_id: str) -> None:
        rds: aioredis.Redis = request.app.state.redis
        key = f"ratelimit:runs:{tenant_id}"
        count = await rds.incr(key)
        if count == 1:
            await rds.expire(key, 60)
        if count > 30:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Per-tenant run rate limit exceeded (30/min)",
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def serve_demo_ui() -> FileResponse:
        index_file = _web_index_path()
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Demo UI not found")
        return FileResponse(index_file)

    @app.post("/v1/dev/token")
    async def dev_token(body: DevTokenRequest) -> dict[str, Any]:
        """Local dev helper for the single-file UI persona switcher."""
        ident = DEV_PERSONAS[body.persona]
        token = mint_user_jwt(ident, secret=cfg.jwt_signing_secret)
        return {
            "token": token,
            "tenant_id": ident.tenant_id,
            "actor_id": ident.actor_id,
            "roles": sorted(ident.roles),
            "langfuse_url": cfg.langfuse_base_url,
        }

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------
    @app.post("/v1/threads", status_code=status.HTTP_201_CREATED)
    async def create_thread(
        request: Request,
        body: CreateThreadRequest,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> dict[str, Any]:
        thread_id = f"thr_{uuid.uuid4().hex[:16]}"
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        async with pool.connection() as conn, conn.transaction():
            await set_tenant_context(conn, caller.tenant_id)
            await conn.execute(
                """
                INSERT INTO app_threads
                    (thread_id, tenant_id, created_by, creator_roles,
                     workflow_name, release_version, checkpoint_compat, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'idle')
                """,
                (
                    thread_id,
                    caller.tenant_id,
                    caller.actor_id,
                    sorted(caller.roles),
                    body.workflow_name,
                    body.release_version,
                    body.checkpoint_compat,
                ),
            )
        return {
            "thread_id": thread_id,
            "tenant_id": caller.tenant_id,
            "created_by": caller.actor_id,
            "status": "idle",
        }

    @app.get("/v1/threads")
    async def list_threads(
        request: Request,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> list[dict[str, Any]]:
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            await cur.execute(
                """
                SELECT thread_id, tenant_id, created_by, workflow_name,
                       release_version, checkpoint_compat, status, created_at, updated_at
                  FROM app_threads
                 WHERE tenant_id = %s
                 ORDER BY updated_at DESC
                 LIMIT 50
                """,
                (caller.tenant_id,),
            )
            rows = await cur.fetchall()
        return [
            {
                "thread_id": r[0],
                "tenant_id": r[1],
                "created_by": r[2],
                "workflow_name": r[3],
                "release_version": r[4],
                "checkpoint_compat": r[5],
                "status": r[6],
                "created_at": r[7].isoformat(),
                "updated_at": r[8].isoformat(),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Runs (Start, Get, Stream SSE, Cancel)
    # ------------------------------------------------------------------
    @app.post("/v1/threads/{thread_id}/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(
        thread_id: str,
        request: Request,
        body: StartRunRequest,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> dict[str, Any]:
        await _enforce_rate_limit(request, caller.tenant_id)
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        rds: aioredis.Redis = request.app.state.redis
        run_id = f"run_{uuid.uuid4().hex[:16]}"

        cancelled_run_ids: list[str] = []
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            await cur.execute(
                "SELECT tenant_id, status FROM app_threads WHERE thread_id = %s FOR UPDATE",
                (thread_id,),
            )
            row = await cur.fetchone()
            if row is None or row[0] != caller.tenant_id:
                raise HTTPException(status_code=404, detail="Thread not found")

            thread_status = row[1]
            if thread_status == "idle" or (
                thread_status == "running" and body.multitask_strategy == "enqueue"
            ):
                pass
            elif thread_status == "running" and body.multitask_strategy == "interrupt":
                # Cancel any waiting queued runs immediately; signal active running run to cancel
                await cur.execute(
                    """
                    UPDATE runs
                       SET status = 'cancelled', cancel_requested = true, updated_at = now()
                     WHERE thread_id = %s AND status = 'queued'
                    """,
                    (thread_id,),
                )
                await cur.execute(
                    """
                    UPDATE runs
                       SET cancel_requested = true, updated_at = now()
                     WHERE thread_id = %s AND status = 'running'
                    RETURNING run_id
                    """,
                    (thread_id,),
                )
                cancelled_run_ids = [r[0] for r in await cur.fetchall()]
            else:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Thread is '{thread_status}'; "
                        f"multitask_strategy='{body.multitask_strategy}' rejected start run"
                    ),
                )

            await cur.execute(
                """
                INSERT INTO runs (run_id, thread_id, tenant_id, kind, input, status)
                VALUES (%s, %s, %s, 'start', %s::jsonb, 'queued')
                """,
                (
                    run_id,
                    thread_id,
                    caller.tenant_id,
                    json.dumps({"request_text": body.request_text}),
                ),
            )
            await cur.execute(
                "UPDATE app_threads SET status = 'running', updated_at = now() "
                "WHERE thread_id = %s",
                (thread_id,),
            )

        for cid in cancelled_run_ids:
            await rds.publish(CANCEL_CHANNEL, cid)
        await rds.publish(WAKEUP_CHANNEL, run_id)
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "status": "queued",
            "multitask_strategy": body.multitask_strategy,
        }

    @app.get("/v1/runs/{run_id}")
    async def get_run(
        run_id: str,
        request: Request,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> dict[str, Any]:
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            await cur.execute(
                """
                SELECT r.run_id, r.thread_id, r.tenant_id, r.kind, r.status,
                       r.attempt, r.cancel_requested, r.error_category, r.error_detail,
                       t.status AS thread_status
                  FROM runs r
                  JOIN app_threads t ON t.thread_id = r.thread_id
                 WHERE r.run_id = %s
                """,
                (run_id,),
            )
            row = await cur.fetchone()
        if row is None or row[2] != caller.tenant_id:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run_id": row[0],
            "thread_id": row[1],
            "tenant_id": row[2],
            "kind": row[3],
            "status": row[4],
            "attempt": row[5],
            "cancel_requested": row[6],
            "error_category": row[7],
            "error_detail": row[8],
            "thread_status": row[9],
        }

    @app.get("/v1/runs/{run_id}/stream")
    async def stream_run(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> StreamingResponse:
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            await cur.execute("SELECT tenant_id FROM runs WHERE run_id = %s", (run_id,))
            row = await cur.fetchone()
        if row is None or row[0] != caller.tenant_id:
            raise HTTPException(status_code=404, detail="Run not found")

        rds: aioredis.Redis = request.app.state.redis
        stream_key = f"run:{run_id}:events"
        cursor_id = last_event_id or "0-0"

        async def _event_gen() -> AsyncIterator[str]:
            nonlocal cursor_id
            while True:
                if await request.is_disconnected():
                    break
                entries = await rds.xread({stream_key: cursor_id}, count=20, block=1000)
                if entries:
                    for _, msgs in entries:
                        for msg_id, fields in msgs:
                            cursor_id = msg_id
                            ev = fields.get("event", "message")
                            data = fields.get("data", "{}")
                            yield f"id: {msg_id}\nevent: {ev}\ndata: {data}\n\n"
                            if ev in {"done", "interrupt", "error"}:
                                return
                else:
                    # Check if run already reached terminal state
                    async with (
                        pool.connection() as conn,
                        conn.transaction(),
                        conn.cursor() as cur,
                    ):
                        await set_tenant_context(conn, caller.tenant_id)
                        await cur.execute("SELECT status FROM runs WHERE run_id = %s", (run_id,))
                        st_row = await cur.fetchone()
                    if st_row and st_row[0] in {
                        "succeeded",
                        "failed",
                        "cancelled",
                        "awaiting_approval",
                    }:
                        yield (
                            f"event: done\ndata: {json.dumps({'status': st_row[0]})}\n\n"
                        )
                        return
                    await asyncio.sleep(0.1)

        return StreamingResponse(_event_gen(), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: str,
        request: Request,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> dict[str, Any]:
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        rds: aioredis.Redis = request.app.state.redis

        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            await cur.execute(
                "SELECT tenant_id, thread_id, status, kind FROM runs WHERE run_id = %s FOR UPDATE",
                (run_id,),
            )
            row = await cur.fetchone()
            if row is None or row[0] != caller.tenant_id:
                raise HTTPException(status_code=404, detail="Run not found")

            thread_id, run_status, run_kind = row[1], row[2], row[3]

            # Atomic cancel-before-claim (AC-14).
            # Only a cancelled queued `start` run returns its thread to `idle`;
            # a cancelled queued `resume` run leaves the thread `cancelled` because
            # its approval was already decided.
            await cur.execute(
                """
                UPDATE runs
                   SET status = 'cancelled', cancel_requested = true, updated_at = now()
                 WHERE run_id = %s AND status = 'queued'
                """,
                (run_id,),
            )
            if cur.rowcount == 1:
                await cur.execute(
                    "SELECT status FROM app_threads WHERE thread_id = %s FOR UPDATE",
                    (thread_id,),
                )
                t_row = await cur.fetchone()
                current_thread_status = t_row[0] if t_row else "idle"

                await cur.execute(
                    """
                    SELECT 1
                      FROM runs
                     WHERE thread_id = %s
                       AND run_id <> %s
                       AND status IN ('running', 'queued')
                     LIMIT 1
                    """,
                    (thread_id, run_id),
                )
                has_other_active_run = (await cur.fetchone()) is not None

                if current_thread_status == "awaiting_approval":
                    next_thread_status = "awaiting_approval"
                elif has_other_active_run:
                    next_thread_status = "running"
                else:
                    next_thread_status = "idle" if run_kind == "start" else "cancelled"

                await cur.execute(
                    "UPDATE app_threads SET status = %s, updated_at = now() "
                    "WHERE thread_id = %s",
                    (next_thread_status, thread_id),
                )
                return {
                    "run_id": run_id,
                    "status": "cancelled",
                    "thread_status": next_thread_status,
                    "cancelled_before_claim": True,
                }

            if run_status == "running":
                await cur.execute(
                    "UPDATE runs SET cancel_requested = true, updated_at = now() "
                    "WHERE run_id = %s",
                    (run_id,),
                )
                await rds.publish(CANCEL_CHANNEL, run_id)
                return {"run_id": run_id, "status": "running", "cancel_requested": True}

            return {"run_id": run_id, "status": run_status, "cancel_requested": False}

    # ------------------------------------------------------------------
    # Approvals (Actionable Inbox + Single-Decision Guarantee)
    # ------------------------------------------------------------------
    @app.get("/v1/approvals")
    async def list_approvals(
        request: Request,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> list[dict[str, Any]]:
        if "ops-approver" not in caller.roles and "admin" not in caller.roles:
            raise HTTPException(status_code=403, detail="ops-approver role required")

        pool: AsyncConnectionPool[Any] = request.app.state.pool
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            # v1.3: only actionable when app_threads.status == 'awaiting_approval'
            await cur.execute(
                """
                SELECT a.approval_id, a.thread_id, a.tenant_id, a.requested_by,
                       a.operation_id, a.plan, a.plan_hash, a.status, a.created_at
                  FROM approvals a
                  JOIN app_threads t ON t.thread_id = a.thread_id
                 WHERE a.tenant_id = %s
                   AND a.status = 'pending'
                   AND t.status = 'awaiting_approval'
                 ORDER BY a.created_at DESC
                """,
                (caller.tenant_id,),
            )
            rows = await cur.fetchall()
        return [
            {
                "approval_id": r[0],
                "thread_id": r[1],
                "tenant_id": r[2],
                "requested_by": r[3],
                "operation_id": r[4],
                "plan": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                "plan_hash": r[6],
                "status": r[7],
                "created_at": r[8].isoformat(),
            }
            for r in rows
        ]

    @app.post("/v1/approvals/{approval_id}/decision", status_code=status.HTTP_202_ACCEPTED)
    async def decide_approval(
        approval_id: str,
        request: Request,
        body: ApprovalDecisionRequest,
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> dict[str, Any]:
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        rds: aioredis.Redis = request.app.state.redis
        resume_run_id = f"run_{uuid.uuid4().hex[:16]}"

        async with pool.connection() as conn, conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            await cur.execute(
                """
                SELECT a.thread_id, a.tenant_id, a.requested_by, a.operation_id,
                       a.plan, a.plan_hash, a.status, t.status AS thread_status,
                       t.release_version
                  FROM approvals a
                  JOIN app_threads t ON t.thread_id = a.thread_id
                 WHERE a.approval_id = %s
                """,
                (approval_id,),
            )
            row = await cur.fetchone()
            if row is None or row[1] != caller.tenant_id:
                raise HTTPException(status_code=404, detail="Approval not found")

            (
                thread_id,
                tenant_id,
                requested_by,
                operation_id,
                orig_plan_raw,
                orig_plan_hash,
                apr_status,
                thread_status,
                release_version,
            ) = row

            authz = policy.check_approval_authority(
                caller=caller,
                approval_tenant_id=tenant_id,
                requested_by=requested_by,
            )
            if authz.decision == PolicyDecision.DENY:
                await write_audit_event(
                    conn,
                    AuditEvent(
                        tenant_id=tenant_id,
                        actor_id=caller.actor_id,
                        requested_by=requested_by,
                        thread_id=thread_id,
                        run_id=resume_run_id,
                        workflow_version=release_version,
                        event_type="approval_refused",
                        decision="refused",
                        step_or_op_key=f"refuse:{approval_id}:{caller.actor_id}",
                        operation_id=operation_id,
                        details={"reason": authz.reason},
                    ),
                )
                await conn.commit()
                raise HTTPException(status_code=403, detail=authz.reason)

            orig_plan = (
                orig_plan_raw if isinstance(orig_plan_raw, dict) else json.loads(orig_plan_raw)
            )
            new_status_map = {"approve": "approved", "edit": "edited", "reject": "rejected"}
            new_status = new_status_map[body.decision]

            final_plan: dict[str, Any] | None = None
            final_plan_hash: str | None = None
            if body.decision == "approve":
                final_plan = orig_plan
                final_plan_hash = orig_plan_hash
            elif body.decision == "edit":
                merged = {**orig_plan, **(body.edits or {})}
                edited_obj = ChangePlan.model_validate(merged)
                final_plan = edited_obj.model_dump(mode="json")
                final_plan_hash = edited_obj.plan_hash()

            # Atomic single-decision + actionable-pause UPDATE (spec §8.1 v1.3)
            await cur.execute(
                """
                UPDATE approvals a
                   SET status = %s,
                       decided_by = %s,
                       decided_at = now(),
                       comment = %s,
                       approved_plan = %s::jsonb,
                       approved_plan_hash = %s
                  FROM app_threads t
                 WHERE a.approval_id = %s
                   AND a.thread_id = t.thread_id
                   AND t.status = 'awaiting_approval'
                   AND a.status = 'pending'
                   AND a.requested_by <> %s
                """,
                (
                    new_status,
                    caller.actor_id,
                    body.comment,
                    json.dumps(final_plan) if final_plan is not None else None,
                    final_plan_hash,
                    approval_id,
                    caller.actor_id,
                ),
            )
            if cur.rowcount == 0:
                await conn.rollback()
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Approval not actionable (approval={apr_status}, thread={thread_status})"
                    ),
                )

            resume_payload: dict[str, Any] = {
                "decision": body.decision,
                "approver_id": caller.actor_id,
                "edits": body.edits or {},
                "comment": body.comment,
                "approval_id": approval_id,
                "approved_plan_hash": final_plan_hash,
            }
            await cur.execute(
                """
                INSERT INTO runs (run_id, thread_id, tenant_id, kind, input, status)
                VALUES (%s, %s, %s, 'resume', %s::jsonb, 'queued')
                """,
                (resume_run_id, thread_id, tenant_id, json.dumps(resume_payload)),
            )
            await cur.execute(
                "UPDATE app_threads SET status = 'running', updated_at = now() "
                "WHERE thread_id = %s",
                (thread_id,),
            )
            await write_audit_event(
                conn,
                AuditEvent(
                    tenant_id=tenant_id,
                    actor_id=caller.actor_id,
                    requested_by=requested_by,
                    approved_by=caller.actor_id,
                    thread_id=thread_id,
                    run_id=resume_run_id,
                    workflow_version=release_version,
                    event_type="approval_decided",
                    decision=new_status,  # type: ignore[arg-type]
                    step_or_op_key=f"decided:{approval_id}",
                    operation_id=operation_id,
                    details={"comment": body.comment, "approved_plan_hash": final_plan_hash},
                ),
            )
            await conn.commit()

        await rds.publish(WAKEUP_CHANNEL, resume_run_id)
        return {
            "run_id": resume_run_id,
            "thread_id": thread_id,
            "approval_id": approval_id,
            "decision": body.decision,
        }

    # ------------------------------------------------------------------
    # Audit Log Query
    # ------------------------------------------------------------------
    @app.get("/v1/audit")
    async def list_audit(
        request: Request,
        thread_id: str | None = Query(default=None),
        caller: CallerIdentity = Depends(get_caller),  # noqa: B008
    ) -> list[dict[str, Any]]:
        pool: AsyncConnectionPool[Any] = request.app.state.pool
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, caller.tenant_id)
            if thread_id:
                await cur.execute(
                    """
                    SELECT event_id, tenant_id, actor_id, requested_by, approved_by,
                           thread_id, run_id, workflow_version, event_type,
                           tool_id, operation_id, decision, details, created_at
                      FROM audit_log
                     WHERE tenant_id = %s AND thread_id = %s
                     ORDER BY created_at DESC
                     LIMIT 100
                    """,
                    (caller.tenant_id, thread_id),
                )
            else:
                await cur.execute(
                    """
                    SELECT event_id, tenant_id, actor_id, requested_by, approved_by,
                           thread_id, run_id, workflow_version, event_type,
                           tool_id, operation_id, decision, details, created_at
                      FROM audit_log
                     WHERE tenant_id = %s
                     ORDER BY created_at DESC
                     LIMIT 100
                    """,
                    (caller.tenant_id,),
                )
            rows = await cur.fetchall()
        return [
            {
                "event_id": str(r[0]),
                "tenant_id": r[1],
                "actor_id": r[2],
                "requested_by": r[3],
                "approved_by": r[4],
                "thread_id": r[5],
                "run_id": r[6],
                "workflow_version": r[7],
                "event_type": r[8],
                "tool_id": r[9],
                "operation_id": r[10],
                "decision": r[11],
                "details": r[12] if isinstance(r[12], dict) else json.loads(r[12]),
                "created_at": r[13].isoformat(),
            }
            for r in rows
        ]

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
