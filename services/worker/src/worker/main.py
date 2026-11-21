"""Separate Worker process: claims runs from Postgres, manages fenced leases, executes graphs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import uuid
from typing import Any

import psycopg
import redis.asyncio as aioredis
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from core.audit import AuditEvent, write_audit_event
from core.auth import mint_mcp_token
from core.config import Settings, get_settings
from core.db import set_system_rls_bypass, set_tenant_context
from core.models import RunContext
from mcp_servers.domain_store import PostgresDomainStore
from telemetry.tracing import (
    build_langchain_callbacks,
    build_trace_attributes,
    configure_tracer_provider,
)
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph
from workflows.incident_desk.version import RELEASE_VERSION
from workflows.mcp_client import call_mcp_tool
from workflows.registry import get_workflow_spec

WAKEUP_CHANNEL = "ops:runs:wakeup"
CANCEL_CHANNEL = "ops:runs:cancel"


def _stream_key(run_id: str) -> str:
    return f"run:{run_id}:events"


def _classify_terminal_state(vals: dict[str, Any]) -> tuple[str, str, str | None, str]:
    """Single source of truth for terminal run/thread status across normal & recovery paths."""
    final_st = str(vals.get("final_status") or "completed")
    err_cat = vals.get("error_category")
    failed_outcome = err_cat in {"fatal", "outcome_unknown"} or (
        final_st in {"denied_plan_mismatch", "verification_failed"}
    )
    run_st = "failed" if failed_outcome else "succeeded"
    thread_st = "failed" if run_st == "failed" else "completed"
    return run_st, thread_st, err_cat, final_st


class RunWorker:
    """Executes LangGraph runs claimed via PostgreSQL `FOR UPDATE SKIP LOCKED`."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        worker_id: str | None = None,
        lease_s: int | None = None,
        heartbeat_s: float | None = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.settings = settings or get_settings()
        self.worker_id = worker_id or f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.lease_s = lease_s or int(
            os.getenv("WORKER_LEASE_S", str(self.settings.worker_lease_s))
        )
        self.heartbeat_s = heartbeat_s or float(
            os.getenv("WORKER_HEARTBEAT_S", str(self.settings.worker_heartbeat_s))
        )
        self.poll_interval_s = poll_interval_s
        self._stop = asyncio.Event()
        self._tracer = configure_tracer_provider(
            settings=self.settings, service_name=f"ops-worker-{self.worker_id}"
        ).get_tracer("worker")

    def request_stop(self) -> None:
        self._stop.set()

    async def _publish_event(
        self,
        rds: aioredis.Redis,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        await rds.xadd(
            _stream_key(run_id),
            {"event": event_type, "data": json.dumps(payload)},
            maxlen=500,
            approximate=True,
        )

    async def _reconcile_in_flight_write(
        self,
        *,
        conn: psycopg.AsyncConnection[Any],
        domain: PostgresDomainStore,
        tenant_id: str,
        requester_id: str,
        thread_id: str,
        run_id: str,
        attempt: int,
        operation_id: str,
        release_version: str,
        mcp_url: str | None = None,
    ) -> tuple[str, str, str | None, dict[str, Any]]:
        """Reconcile an in-flight MCP write before settling a cancelled/expired run."""
        try:
            if self.settings.mcp_transport == "http":
                tok = mint_mcp_token(
                    tenant_id=tenant_id,
                    requester_id=requester_id,
                    run_id=run_id,
                    operation_id=operation_id,
                    secret=self.settings.mcp_token_secret,
                )
                rec = await asyncio.wait_for(
                    call_mcp_tool(
                        url=mcp_url or self.settings.mcp_infra_url,
                        bearer_token=tok,
                        tool_name="get_operation_status",
                        arguments={"operation_id": operation_id},
                    ),
                    timeout=self.settings.tool_timeout_s,
                )
            else:
                rec = await asyncio.wait_for(
                    domain.get_operation_status(tenant_id=tenant_id, operation_id=operation_id),
                    timeout=self.settings.tool_timeout_s,
                )
        except Exception as exc:
            rec = {
                "operation_id": operation_id,
                "status": "outcome_unknown",
                "error": str(exc),
            }

        if rec.get("status") == "succeeded":
            decision = "succeeded"
            run_status = "succeeded"
            thread_status = "completed"
            err_cat: str | None = None
        elif rec.get("status") == "failed":
            decision = "failed"
            run_status = "failed"
            thread_status = "failed"
            err_cat = "fatal"
        else:
            decision = "outcome_unknown"
            run_status = "failed"
            thread_status = "failed"
            err_cat = "outcome_unknown"

        await write_audit_event(
            conn,
            AuditEvent(
                tenant_id=tenant_id,
                actor_id=requester_id,
                requested_by=requester_id,
                thread_id=thread_id,
                run_id=run_id,
                workflow_version=release_version,
                event_type="reconciliation",
                tool_id="get_operation_status",
                operation_id=operation_id,
                decision=decision,  # type: ignore[arg-type]
                step_or_op_key=f"cancel_reconcile:{operation_id}",
                attempt=attempt,
                details=rec,
            ),
        )
        return run_status, thread_status, err_cat, rec

    async def _sweep_expired_runs(
        self,
        pool: AsyncConnectionPool[Any],
        rds: aioredis.Redis,
        graph: Any,
        domain: PostgresDomainStore,
    ) -> None:
        """Finalize expired runs that exhausted max_attempts or had cancel_requested=true."""
        async with pool.connection() as conn:
            await set_system_rls_bypass(conn)
            async with conn.cursor() as cur:
                # 1. Exhausted attempts
                await cur.execute(
                    """
                    UPDATE runs
                       SET status = 'failed',
                           error_category = 'transient',
                           error_detail = 'max_attempts_exhausted',
                           lease_owner = NULL,
                           updated_at = now()
                     WHERE status = 'running'
                       AND lease_expires_at < now()
                       AND attempt >= max_attempts
                    RETURNING run_id, thread_id, tenant_id, attempt
                    """
                )
                exhausted = await cur.fetchall()
                for run_id, thread_id, tenant_id, attempt in exhausted:
                    await cur.execute(
                        "UPDATE app_threads SET status = 'failed', updated_at = now() "
                        "WHERE thread_id = %s",
                        (thread_id,),
                    )
                    await write_audit_event(
                        conn,
                        AuditEvent(
                            tenant_id=tenant_id,
                            actor_id="system",
                            thread_id=thread_id,
                            run_id=run_id,
                            workflow_version=RELEASE_VERSION,
                            event_type="run_reclaimed",
                            decision="failed",
                            step_or_op_key=f"exhausted:{run_id}",
                            attempt=attempt,
                            details={"reason": "max_attempts_exhausted"},
                        ),
                    )
                    await self._publish_event(
                        rds, run_id, "error", {"error": "max_attempts_exhausted"}
                    )

                # 2. Expired + cancel_requested -> reconcile if write may have started
                await cur.execute(
                    """
                    SELECT r.run_id, r.thread_id, r.tenant_id, r.attempt,
                           t.created_by, t.release_version, t.workflow_name
                      FROM runs r
                      JOIN app_threads t ON t.thread_id = r.thread_id
                     WHERE r.status = 'running'
                       AND r.lease_expires_at < now()
                       AND r.cancel_requested = true
                       FOR UPDATE OF r SKIP LOCKED
                    """
                )
                cancelled_expired = await cur.fetchall()
                for (
                    run_id,
                    thread_id,
                    tenant_id,
                    attempt,
                    created_by,
                    rel_ver,
                    workflow_name,
                ) in cancelled_expired:
                    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
                    op_id = (snap.values or {}).get("operation_id") if snap else None
                    if op_id:
                        try:
                            reconcile_url = get_workflow_spec(workflow_name).reconcile_mcp_url(
                                self.settings
                            )
                        except KeyError:
                            reconcile_url = self.settings.mcp_infra_url
                        r_status, t_status, err_cat, rec = await self._reconcile_in_flight_write(
                            conn=conn,
                            domain=domain,
                            tenant_id=tenant_id,
                            requester_id=created_by,
                            thread_id=thread_id,
                            run_id=run_id,
                            attempt=attempt,
                            operation_id=op_id,
                            release_version=rel_ver,
                            mcp_url=reconcile_url,
                        )
                        detail = json.dumps({"reconciled": rec})
                    else:
                        r_status, t_status, err_cat, detail = (
                            "cancelled",
                            "cancelled",
                            None,
                            "cancelled_expired",
                        )

                    await cur.execute(
                        """
                        UPDATE runs
                           SET status = %s,
                               error_category = %s,
                               error_detail = %s,
                               lease_owner = NULL,
                               updated_at = now()
                         WHERE run_id = %s
                        """,
                        (r_status, err_cat, detail, run_id),
                    )
                    await cur.execute(
                        "UPDATE app_threads SET status = %s, updated_at = now() "
                        "WHERE thread_id = %s",
                        (t_status, thread_id),
                    )
                    await self._publish_event(
                        rds, run_id, "done", {"status": r_status, "detail": detail}
                    )
            await conn.commit()

    async def _claim_one_run(self, pool: AsyncConnectionPool[Any]) -> dict[str, Any] | None:
        async with pool.connection() as conn:
            await set_system_rls_bypass(conn)
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        WITH candidate AS (
                            SELECT r.run_id
                              FROM runs r
                              JOIN app_threads t ON t.thread_id = r.thread_id
                             WHERE (
                                     r.status = 'queued'
                                  OR (r.status = 'running' AND r.lease_expires_at < now())
                                   )
                               AND r.cancel_requested = false
                               AND r.attempt < r.max_attempts
                               AND NOT (r.kind = 'start' AND t.status = 'awaiting_approval')
                               AND NOT (
                                   r.kind = 'start'
                                   AND EXISTS (
                                       SELECT 1
                                         FROM runs r_res
                                        WHERE r_res.thread_id = r.thread_id
                                          AND r_res.kind = 'resume'
                                          AND r_res.status IN ('queued', 'running')
                                   )
                               )
                               AND NOT EXISTS (
                                   SELECT 1
                                     FROM runs r2
                                    WHERE r2.thread_id = r.thread_id
                                      AND r2.status = 'running'
                                      AND r2.run_id <> r.run_id
                               )
                             ORDER BY (CASE WHEN r.kind = 'resume' THEN 0 ELSE 1 END),
                                      r.created_at ASC
                             LIMIT 1
                               FOR UPDATE OF r, t SKIP LOCKED
                        )
                        UPDATE runs r
                           SET status = 'running',
                               lease_owner = %s,
                               lease_expires_at = now() + (%s * interval '1 second'),
                               attempt = r.attempt + 1,
                               updated_at = now()
                          FROM candidate c
                         WHERE r.run_id = c.run_id
                        RETURNING r.run_id, r.thread_id, r.tenant_id, r.kind, r.input, r.attempt
                        """,
                        (self.worker_id, self.lease_s),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return None

                    run_id, thread_id, tenant_id, kind, run_input, attempt = row
                    await cur.execute(
                        """
                        SELECT created_by, creator_roles, release_version, checkpoint_compat,
                               workflow_name
                          FROM app_threads
                         WHERE thread_id = %s
                        """,
                        (thread_id,),
                    )
                    t_row = await cur.fetchone()
                    assert t_row is not None
                    created_by, creator_roles, release_version, thread_compat, workflow_name = (
                        t_row
                    )

                    if attempt > 1:
                        await write_audit_event(
                            conn,
                            AuditEvent(
                                tenant_id=tenant_id,
                                actor_id=created_by,
                                thread_id=thread_id,
                                run_id=run_id,
                                workflow_version=release_version,
                                event_type="run_reclaimed",
                                decision="allow",
                                step_or_op_key=f"reclaim:{run_id}",
                                attempt=attempt,
                                details={"new_worker": self.worker_id},
                            ),
                        )
                await conn.commit()
            except psycopg.errors.UniqueViolation:
                await conn.rollback()
                return None

        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "kind": kind,
            "input": run_input if isinstance(run_input, dict) else json.loads(run_input),
            "attempt": attempt,
            "created_by": created_by,
            "creator_roles": frozenset(creator_roles),
            "release_version": release_version,
            "checkpoint_compat": thread_compat,
            "workflow_name": workflow_name,
        }

    async def _fenced_finalize(
        self,
        pool: AsyncConnectionPool[Any],
        *,
        run_id: str,
        thread_id: str,
        attempt: int,
        run_status: str,
        thread_status: str,
        error_category: str | None = None,
        error_detail: str | None = None,
    ) -> bool:
        """Fenced DB transition: only commits if this worker still holds (lease_owner, attempt)."""
        async with pool.connection() as conn:
            await set_system_rls_bypass(conn)
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE runs
                       SET status = %s,
                           error_category = %s,
                           error_detail = %s,
                           lease_owner = NULL,
                           lease_expires_at = NULL,
                           updated_at = now()
                     WHERE run_id = %s
                       AND lease_owner = %s
                       AND attempt = %s
                    """,
                    (
                        run_status,
                        error_category,
                        error_detail,
                        run_id,
                        self.worker_id,
                        attempt,
                    ),
                )
                if cur.rowcount == 0:
                    await conn.rollback()
                    return False
                # If a terminal run finishes while a follow-up run is queued for this thread
                # (enqueue/interrupt), keep the thread `running` so the queued run can be claimed.
                # Never overwrite `awaiting_approval`, or a queued follow-up would strand the HITL
                # approval and hide it from `GET /v1/approvals`.
                if thread_status == "awaiting_approval":
                    effective_thread_status = "awaiting_approval"
                else:
                    await cur.execute(
                        "SELECT 1 FROM runs WHERE thread_id = %s AND status = 'queued' LIMIT 1",
                        (thread_id,),
                    )
                    has_queued_followup = (await cur.fetchone()) is not None
                    effective_thread_status = "running" if has_queued_followup else thread_status
                await cur.execute(
                    "UPDATE app_threads SET status = %s, updated_at = now() WHERE thread_id = %s",
                    (effective_thread_status, thread_id),
                )
            await conn.commit()
        return True

    async def _is_cancel_requested(self, pool: AsyncConnectionPool[Any], run_id: str) -> bool:
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await set_system_rls_bypass(conn)
            await cur.execute("SELECT cancel_requested FROM runs WHERE run_id = %s", (run_id,))
            row = await cur.fetchone()
            return bool(row and row[0])

    async def _heartbeat_loop(
        self,
        pool: AsyncConnectionPool[Any],
        *,
        run_id: str,
        attempt: int,
        lost_lease: asyncio.Event,
    ) -> None:
        while not lost_lease.is_set():
            await asyncio.sleep(self.heartbeat_s)
            async with pool.connection() as conn:
                await set_system_rls_bypass(conn)
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        UPDATE runs
                           SET lease_expires_at = now() + (%s * interval '1 second'),
                               updated_at = now()
                         WHERE run_id = %s
                           AND lease_owner = %s
                           AND attempt = %s
                           AND status = 'running'
                        """,
                        (self.lease_s, run_id, self.worker_id, attempt),
                    )
                    if cur.rowcount == 0:
                        lost_lease.set()
                        return
                await conn.commit()

    async def _execute_claimed_run(
        self,
        pool: AsyncConnectionPool[Any],
        rds: aioredis.Redis,
        checkpointer: AsyncPostgresSaver,
        store: AsyncPostgresStore,
        claimed: dict[str, Any],
    ) -> None:
        run_id: str = claimed["run_id"]
        thread_id: str = claimed["thread_id"]
        tenant_id: str = claimed["tenant_id"]
        attempt: int = claimed["attempt"]
        kind: str = claimed["kind"]
        run_input: dict[str, Any] = claimed["input"]
        workflow_name: str = claimed.get("workflow_name", "incident_desk")

        try:
            wf_spec = get_workflow_spec(workflow_name)
        except KeyError as exc:
            committed = await self._fenced_finalize(
                pool,
                run_id=run_id,
                thread_id=thread_id,
                attempt=attempt,
                run_status="failed",
                thread_status="failed",
                error_category="fatal",
                error_detail=str(exc),
            )
            if committed:
                await self._publish_event(rds, run_id, "error", {"error": str(exc)})
            return

        # 1. Checkpoint compatibility check (AC-6)
        current_compat = wf_spec.get_checkpoint_compat()
        if claimed["checkpoint_compat"] != current_compat:
            async with pool.connection() as conn:
                await set_tenant_context(conn, tenant_id)
                await write_audit_event(
                    conn,
                    AuditEvent(
                        tenant_id=tenant_id,
                        actor_id=claimed["created_by"],
                        thread_id=thread_id,
                        run_id=run_id,
                        workflow_version=wf_spec.release_version,
                        event_type="version_blocked",
                        decision="deny",
                        step_or_op_key=f"version:{run_id}",
                        attempt=attempt,
                        details={
                            "thread_compat": claimed["checkpoint_compat"],
                            "worker_compat": current_compat,
                        },
                    ),
                )
                await conn.commit()
            committed = await self._fenced_finalize(
                pool,
                run_id=run_id,
                thread_id=thread_id,
                attempt=attempt,
                run_status="failed",
                thread_status="blocked_version",
                error_category="fatal",
                error_detail="migration_required",
            )
            if committed:
                await self._publish_event(rds, run_id, "error", {"error": "migration_required"})
            return

        domain = PostgresDomainStore(self.settings.database_url)
        deps = WorkflowDeps(
            settings=self.settings,
            domain_store=domain,
            use_postgres=True,
            use_mcp_http=(self.settings.mcp_transport == "http"),
            attempt=attempt,
        )
        graph = wf_spec.build_graph(deps=deps, checkpointer=checkpointer, store=store)

        ctx = RunContext(
            tenant_id=tenant_id,
            requester_id=claimed["created_by"],
            requester_roles=claimed["creator_roles"],
            thread_id=thread_id,
            run_id=run_id,
            request_id=f"req-{run_id}",
        )
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.max_graph_steps,
        }
        callbacks = build_langchain_callbacks(self.settings)
        if callbacks:
            config["callbacks"] = callbacks

        lost_lease = asyncio.Event()
        hb_task = asyncio.create_task(
            self._heartbeat_loop(pool, run_id=run_id, attempt=attempt, lost_lease=lost_lease)
        )

        span_attrs = build_trace_attributes(ctx, workflow_version=wf_spec.release_version)
        try:
            with self._tracer.start_as_current_span(f"{wf_spec.name}.run", attributes=span_attrs):
                snap = await graph.aget_state(config)
                has_values = bool(snap and snap.values)
                has_interrupts = bool(snap and snap.interrupts)
                has_next = bool(snap and snap.next)

                # Decide continuation path (spec §8.2 + v2.3 enqueue/interrupt)
                if attempt > 1 and has_interrupts and kind == "start":
                    committed = await self._fenced_finalize(
                        pool,
                        run_id=run_id,
                        thread_id=thread_id,
                        attempt=attempt,
                        run_status="awaiting_approval",
                        thread_status="awaiting_approval",
                    )
                    if committed:
                        await self._publish_event(
                            rds, run_id, "interrupt", snap.interrupts[0].value
                        )
                    return

                if attempt > 1 and not has_next and has_values and not has_interrupts:
                    vals = snap.values if snap else {}
                    run_st, thread_st, err_cat, final_st = _classify_terminal_state(vals)
                    committed = await self._fenced_finalize(
                        pool,
                        run_id=run_id,
                        thread_id=thread_id,
                        attempt=attempt,
                        run_status=run_st,
                        thread_status=thread_st,
                        error_category=err_cat,
                        error_detail=final_st,
                    )
                    if committed:
                        await self._publish_event(
                            rds,
                            run_id,
                            "done",
                            {"status": run_st, "final_status": final_st},
                        )
                    return

                graph_input: Any
                if has_interrupts and kind == "resume":
                    graph_input = Command(resume=run_input)
                elif attempt > 1 and has_values and has_next:
                    graph_input = None  # Crash recovery from last checkpoint
                else:
                    graph_input = run_input

                async def _run_stream() -> None:
                    async for update in graph.astream(
                        graph_input,
                        config=config,
                        context=ctx,
                        stream_mode="updates",
                        durability="sync",
                    ):
                        if lost_lease.is_set():
                            return
                        for node_name in update:
                            await self._publish_event(
                                rds, run_id, "node_update", {"node": node_name}
                            )
                        if await self._is_cancel_requested(pool, run_id):
                            raise asyncio.CancelledError("cancel_requested")

                await asyncio.wait_for(_run_stream(), timeout=self.settings.run_timeout_s)

                if lost_lease.is_set():
                    return

                end_snap = await graph.aget_state(config)
                if end_snap and end_snap.interrupts:
                    committed = await self._fenced_finalize(
                        pool,
                        run_id=run_id,
                        thread_id=thread_id,
                        attempt=attempt,
                        run_status="awaiting_approval",
                        thread_status="awaiting_approval",
                    )
                    if committed:
                        await self._publish_event(
                            rds, run_id, "interrupt", end_snap.interrupts[0].value
                        )
                else:
                    vals = end_snap.values if end_snap else {}
                    run_st, thread_st, err_cat, final_st = _classify_terminal_state(vals)
                    committed = await self._fenced_finalize(
                        pool,
                        run_id=run_id,
                        thread_id=thread_id,
                        attempt=attempt,
                        run_status=run_st,
                        thread_status=thread_st,
                        error_category=err_cat,
                        error_detail=final_st,
                    )
                    if committed:
                        await self._publish_event(
                            rds,
                            run_id,
                            "done",
                            {
                                "status": run_st,
                                "final_status": final_st,
                                "verification_passed": vals.get("verification_passed"),
                                "operation_id": vals.get("operation_id"),
                            },
                        )
        except asyncio.CancelledError:
            # Cancel-after-write reconciliation (AC-16)
            post_snap = await graph.aget_state(config)
            op_id = (post_snap.values or {}).get("operation_id") if post_snap else None
            if op_id:
                async with pool.connection() as conn:
                    await set_tenant_context(conn, tenant_id)
                    r_st, t_st, err_cat, rec = await self._reconcile_in_flight_write(
                        conn=conn,
                        domain=domain,
                        tenant_id=tenant_id,
                        requester_id=claimed["created_by"],
                        thread_id=thread_id,
                        run_id=run_id,
                        attempt=attempt,
                        operation_id=op_id,
                        release_version=claimed["release_version"],
                        mcp_url=wf_spec.reconcile_mcp_url(self.settings),
                    )
                    await conn.commit()
                committed = await self._fenced_finalize(
                    pool,
                    run_id=run_id,
                    thread_id=thread_id,
                    attempt=attempt,
                    run_status=r_st,
                    thread_status=t_st,
                    error_category=err_cat,
                    error_detail=json.dumps({"reconciled": rec}),
                )
                if committed:
                    await self._publish_event(
                        rds, run_id, "done", {"status": r_st, "reconciled": rec}
                    )
            else:
                committed = await self._fenced_finalize(
                    pool,
                    run_id=run_id,
                    thread_id=thread_id,
                    attempt=attempt,
                    run_status="cancelled",
                    thread_status="cancelled",
                )
                if committed:
                    await self._publish_event(rds, run_id, "done", {"status": "cancelled"})
        except TimeoutError:
            committed = await self._fenced_finalize(
                pool,
                run_id=run_id,
                thread_id=thread_id,
                attempt=attempt,
                run_status="failed",
                thread_status="failed",
                error_category="transient",
                error_detail="run_timeout",
            )
            if committed:
                await self._publish_event(rds, run_id, "error", {"error": "run_timeout"})
        except Exception as exc:
            committed = await self._fenced_finalize(
                pool,
                run_id=run_id,
                thread_id=thread_id,
                attempt=attempt,
                run_status="failed",
                thread_status="failed",
                error_category="fatal",
                error_detail=str(exc),
            )
            if committed:
                await self._publish_event(rds, run_id, "error", {"error": str(exc)})
        finally:
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb_task

    async def serve_forever(self) -> None:
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            conninfo=self.settings.database_url,
            min_size=2,
            max_size=10,
            open=False,
        )
        await pool.open()
        rds = aioredis.from_url(self.settings.redis_url, decode_responses=True)
        pubsub = rds.pubsub()
        await pubsub.subscribe(WAKEUP_CHANNEL)

        try:
            async with (
                AsyncPostgresSaver.from_conn_string(self.settings.database_url) as checkpointer,
                AsyncPostgresStore.from_conn_string(self.settings.database_url) as store,
            ):
                domain = PostgresDomainStore(self.settings.database_url)
                sweep_graph = build_incident_desk_graph(
                    deps=WorkflowDeps(
                        settings=self.settings, domain_store=domain, use_postgres=True
                    ),
                    checkpointer=checkpointer,
                    store=store,
                )
                while not self._stop.is_set():
                    await self._sweep_expired_runs(pool, rds, sweep_graph, domain)
                    claimed = await self._claim_one_run(pool)
                    if claimed is not None:
                        await self._execute_claimed_run(pool, rds, checkpointer, store, claimed)
                        continue
                    # Wait briefly for Redis wake-up or poll fallback
                    try:
                        await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=self.poll_interval_s
                        )
                    except Exception:
                        await asyncio.sleep(self.poll_interval_s)
        finally:
            await pubsub.close()
            await rds.aclose()
            await pool.close()


def run() -> None:
    worker = RunWorker()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, worker.request_stop)
    try:
        loop.run_until_complete(worker.serve_forever())
    finally:
        loop.close()
