"""Shared governed workflow lifecycle for LangGraph StateGraphs (incident_desk, refund_ops).

Preserves exact checkpoint node names and state keys across all workflows while allowing each
domain to supply only its read tools, prompts, plan synthesis, write execution, and verification.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import psycopg
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from core.audit import AuditEvent, write_audit_event
from core.config import Settings, get_settings
from core.db import set_tenant_context
from core.models import ChangePlan, RunContext
from core.pii import API_KEY_PATTERN, EMAIL_PATTERN, PHONE_PATTERN, redact_obj, redact_text
from core.policy import PolicyDecision, PolicyEngine
from core.workflow_catalog import PROPOSE_PLAN_PROMPT_TEMPLATE
from gateway.factory import get_chat_model, get_embeddings
from knowledge.fixtures import SEED_RUNBOOKS
from knowledge.retriever import retrieve_runbooks
from mcp_servers.domain_store import DomainStore, InMemoryDomainStore, PostgresDomainStore
from workflows.incident_desk.state import IncidentDeskState
from workflows.tool_executor import ToolExecutor, WriteSpec


def maybe_inject_fault(point: str) -> None:
    """If FAULT_INJECT matches `point`, crash immediately (same as SIGKILL)."""
    if os.getenv("FAULT_INJECT") == point:
        os._exit(137)


def coerce_plan(raw: ChangePlan | dict[str, Any] | None) -> ChangePlan | None:
    if raw is None:
        return None
    if isinstance(raw, ChangePlan):
        return raw
    return ChangePlan.model_validate(raw)


@dataclass
class WorkflowDeps:
    """Injectable dependencies so the same StateGraph runs in Studio, pytest, and Worker."""

    settings: Settings = field(default_factory=get_settings)
    domain_store: DomainStore | None = None
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    use_postgres: bool = False
    use_mcp_http: bool = False
    approvals_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_memory: list[AuditEvent] = field(default_factory=list)
    attempt: int = 1
    simulate_write_timeout: bool = False

    def get_domain_store(self) -> DomainStore:
        if self.domain_store is not None:
            return self.domain_store
        if self.use_postgres:
            return PostgresDomainStore(self.settings.database_url)
        self.domain_store = InMemoryDomainStore()
        return self.domain_store


class StudioRunContext(RunContext, frozen=True):
    """Dev fixture defaults so `langgraph dev` Studio works with an empty context payload."""

    tenant_id: str = "tenant-alpha"
    requester_id: str = "alice"
    requester_roles: frozenset[str] = frozenset({"engineer"})
    thread_id: str = "studio-thread"
    run_id: str = "studio-run"
    request_id: str = "studio-req"


@dataclass(frozen=True)
class DomainHooks:
    """Domain-specific hooks plugged into the shared governed workflow lifecycle."""

    workflow_name: str
    release_version: str
    investigate_system_prompt: str
    build_read_tools: Callable[[ToolExecutor], list[Any]]
    build_mock_investigate_turns: Callable[
        [str, RunContext, list[dict[str, Any]]], list[AIMessage]
    ]
    deterministic_mock_plan: Callable[[str, list[dict[str, Any]]], ChangePlan]
    resolve_write_spec: Callable[[ToolExecutor, RunContext, ChangePlan, str], WriteSpec]
    verify_domain_end_state: Callable[
        [ToolExecutor, RunContext, ChangePlan, str, dict[str, Any]],
        Awaitable[bool],
    ]
    build_memory_payload: Callable[[RunContext, ChangePlan, str], dict[str, Any]]


def build_governed_workflow_graph(
    hooks: DomainHooks,
    *,
    deps: WorkflowDeps | None = None,
    checkpointer: Any = None,
    store: Any = None,
    context_schema: type[RunContext] = RunContext,
    default_context: RunContext | None = None,
):
    """Compile a 12-node governed StateGraph sharing retrieval, HITL, policy, audit, & recovery."""
    wf_deps = deps or WorkflowDeps()
    domain = wf_deps.get_domain_store()
    executor = ToolExecutor(
        settings=wf_deps.settings,
        domain_store=domain,
        use_mcp_http=wf_deps.use_mcp_http,
    )
    read_only_tools = hooks.build_read_tools(executor)

    def _ctx(runtime: Runtime[RunContext]) -> RunContext:
        if runtime.context is not None:
            return runtime.context
        if default_context is not None:
            return default_context
        raise ValueError("Missing required RunContext")

    async def _record_audit(event: AuditEvent) -> None:
        if wf_deps.use_postgres:
            async with (
                await psycopg.AsyncConnection.connect(wf_deps.settings.database_url) as conn,
                conn.transaction(),
            ):
                await set_tenant_context(conn, event.tenant_id)
                await write_audit_event(conn, event)
        else:
            wf_deps.audit_memory.append(event)

    # Node 1: retrieve_context (ACL pgvector + tenant-scoped AsyncPostgresStore memory)
    async def retrieve_context(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        query = state.get("request_text", "")
        if wf_deps.use_postgres:
            embeddings = get_embeddings(tenant_id=ctx.tenant_id, settings=wf_deps.settings)
            async with await psycopg.AsyncConnection.connect(
                wf_deps.settings.database_url
            ) as conn:
                docs = await retrieve_runbooks(
                    conn,
                    embeddings=embeddings,
                    tenant_id=ctx.tenant_id,
                    roles=ctx.requester_roles,
                    query=query,
                    limit=4,
                )
            runbooks = [d.model_dump() for d in docs]
        else:
            runbooks = [
                {
                    "doc_id": rb.doc_id,
                    "title": rb.title,
                    "content": rb.content,
                    "metadata": rb.metadata,
                }
                for rb in SEED_RUNBOOKS
                if rb.tenant_id == ctx.tenant_id
                and (set(rb.allowed_roles) & set(ctx.requester_roles))
            ]

        prior_memories: list[dict[str, Any]] = []
        if runtime.store is not None:
            mem_items = await runtime.store.asearch(
                (ctx.tenant_id, "memory", hooks.workflow_name),
                limit=5,
            )
            prior_memories = [dict(item.value) for item in mem_items]

        return {
            "release_version": hooks.release_version,
            "retrieved_runbooks": runbooks,
            "prior_memories": prior_memories,
            # Reset per-run fields so reused threads never leak stale state from prior runs
            "investigation_summary": "",
            "change_plan": None,
            "policy_decision": None,
            "operation_id": None,
            "approval_decision": None,
            "approved_plan_hash": None,
            "execution_result": None,
            "verification_passed": None,
            "final_status": None,
            "error_category": None,
        }

    # Node 2: investigate (LangChain v1 create_agent + Middleware, read-only tools ONLY)
    async def investigate(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        maybe_inject_fault("investigate")
        ctx = _ctx(runtime)
        req_text = state.get("request_text", "")
        runbooks = state.get("retrieved_runbooks", [])

        if wf_deps.settings.model_mode == "mock":
            mock_turns = hooks.build_mock_investigate_turns(req_text, ctx, runbooks)
            chat_model = get_chat_model(
                alias="ops-primary",
                tenant_id=ctx.tenant_id,
                settings=wf_deps.settings,
                mock_responses=mock_turns,
            )
            fast_model = get_chat_model(
                alias="ops-fast",
                tenant_id=ctx.tenant_id,
                settings=wf_deps.settings,
                mock_responses=[AIMessage(content="Summary")],
            )
        else:
            chat_model = get_chat_model(
                alias="ops-primary", tenant_id=ctx.tenant_id, settings=wf_deps.settings
            )
            fast_model = get_chat_model(
                alias="ops-fast", tenant_id=ctx.tenant_id, settings=wf_deps.settings
            )

        agent = create_agent(
            model=chat_model,
            tools=read_only_tools,
            system_prompt=hooks.investigate_system_prompt,
            context_schema=RunContext,
            middleware=[
                PIIMiddleware("email", detector=EMAIL_PATTERN, strategy="redact"),
                PIIMiddleware("phone_number", detector=PHONE_PATTERN, strategy="redact"),
                PIIMiddleware("api_key", detector=API_KEY_PATTERN, strategy="redact"),
                SummarizationMiddleware(
                    model=fast_model,
                    trigger=("tokens", 4000),
                    keep=("messages", 10),
                ),
                ModelCallLimitMiddleware(run_limit=wf_deps.settings.max_model_calls),
                ToolCallLimitMiddleware(run_limit=wf_deps.settings.max_tool_calls),
            ],
        )

        agent_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": req_text}]},
            context=ctx,
        )
        last_msg = agent_result["messages"][-1]
        summary = getattr(last_msg, "content", str(last_msg))
        return {
            "messages": agent_result["messages"],
            "investigation_summary": str(summary),
        }

    # Node 3: propose_plan (Structured ChangePlan)
    async def propose_plan(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        runbooks = state.get("retrieved_runbooks", [])
        req_text = state.get("request_text", "")

        if wf_deps.settings.model_mode == "mock":
            plan = hooks.deterministic_mock_plan(req_text, runbooks)
        else:
            model = get_chat_model(
                alias="ops-primary", tenant_id=ctx.tenant_id, settings=wf_deps.settings
            )
            structured = model.with_structured_output(ChangePlan)
            safe_req = redact_text(req_text)
            safe_summary = redact_text(state.get("investigation_summary", ""))
            safe_runbooks = redact_obj(runbooks)
            plan = await structured.ainvoke(
                PROPOSE_PLAN_PROMPT_TEMPLATE.format(
                    request=safe_req,
                    investigation=safe_summary,
                    runbooks=json.dumps(safe_runbooks),
                )
            )

        return {"change_plan": plan.model_dump(mode="json")}

    def route_after_plan(state: IncidentDeskState) -> str:
        plan = coerce_plan(state.get("change_plan"))
        if plan is None or plan.action == "none":
            return "finalize_no_action"
        return "evaluate_policy"

    # Node 4: evaluate_policy + durable operation_id generation
    async def evaluate_policy(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = coerce_plan(state.get("change_plan"))
        assert plan is not None

        res = wf_deps.policy.evaluate_change_plan(ctx=ctx, plan=plan)
        operation_id = f"op_{uuid.uuid4().hex[:16]}"
        audit_decision = "deny" if res.decision == PolicyDecision.DENY else "allow"
        details: dict[str, Any] = {
            "reason": res.reason,
            "action": plan.action,
            "target": plan.target_resource,
        }
        if plan.amount_cents is not None:
            details["amount_cents"] = plan.amount_cents

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=ctx.requester_id,
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", hooks.release_version),
                event_type="authz_check",
                decision=audit_decision,
                step_or_op_key=f"policy:{operation_id}",
                attempt=wf_deps.attempt,
                operation_id=operation_id,
                details=details,
            )
        )
        if res.decision == PolicyDecision.DENY:
            return {
                "policy_decision": "deny",
                "operation_id": operation_id,
                "error_category": "permission_denied",
            }
        return {
            "policy_decision": "require_approval",
            "operation_id": operation_id,
        }

    def route_after_policy(state: IncidentDeskState) -> str:
        if state.get("policy_decision") == "deny":
            return "finalize_denied"
        return "record_approval_request"

    # Node 5: record_approval_request (Idempotent; NO interrupt() here)
    async def record_approval_request(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id")
        assert plan is not None and operation_id is not None

        approval_id = f"apr_{operation_id}"
        plan_json = plan.model_dump(mode="json")
        plan_hash = plan.plan_hash()

        if wf_deps.use_postgres:
            async with (
                await psycopg.AsyncConnection.connect(wf_deps.settings.database_url) as conn,
                conn.transaction(),
            ):
                await set_tenant_context(conn, ctx.tenant_id)
                await conn.execute(
                    """
                    INSERT INTO approvals
                        (approval_id, thread_id, tenant_id, requested_by,
                         operation_id, plan, plan_hash, status)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, 'pending')
                    ON CONFLICT (approval_id) DO NOTHING
                    """,
                    (
                        approval_id,
                        ctx.thread_id,
                        ctx.tenant_id,
                        ctx.requester_id,
                        operation_id,
                        json.dumps(plan_json),
                        plan_hash,
                    ),
                )
        else:
            wf_deps.approvals_memory.setdefault(
                approval_id,
                {
                    "approval_id": approval_id,
                    "thread_id": ctx.thread_id,
                    "tenant_id": ctx.tenant_id,
                    "requested_by": ctx.requester_id,
                    "operation_id": operation_id,
                    "plan": plan_json,
                    "plan_hash": plan_hash,
                    "status": "pending",
                },
            )

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=ctx.requester_id,
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", hooks.release_version),
                event_type="approval_requested",
                decision="allow",
                step_or_op_key=approval_id,
                attempt=wf_deps.attempt,
                operation_id=operation_id,
                details={"approval_id": approval_id, "plan_hash": plan_hash},
            )
        )
        return {}

    # Node 6: await_approval (ONLY interrupt() — safe for replay)
    async def await_approval(state: IncidentDeskState) -> dict[str, Any]:
        plan = coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id")
        decision_payload = interrupt(
            {
                "approval_id": f"apr_{operation_id}",
                "operation_id": operation_id,
                "plan": plan.model_dump(mode="json") if plan else None,
                "plan_hash": plan.plan_hash() if plan else None,
            }
        )
        updates: dict[str, Any] = {"approval_decision": decision_payload}
        if isinstance(decision_payload, dict) and decision_payload.get("decision") == "approve":
            updates["approved_plan_hash"] = decision_payload.get("approved_plan_hash")
        return updates

    def route_after_approval(state: IncidentDeskState) -> str:
        dec = (state.get("approval_decision") or {}).get("decision")
        if dec == "approve":
            return "execute_change"
        if dec == "edit":
            return "revalidate_edit"
        return "finalize_rejected"

    # Node 7: revalidate_edit (Re-run PolicyEngine + new operation_id)
    async def revalidate_edit(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = coerce_plan(state.get("change_plan"))
        decision = state.get("approval_decision") or {}
        edits = decision.get("edits") or {}
        assert plan is not None

        merged = {**plan.model_dump(mode="json"), **edits}
        edited_plan = ChangePlan.model_validate(merged)
        res = wf_deps.policy.evaluate_change_plan(ctx=ctx, plan=edited_plan)
        new_op_id = f"op_{uuid.uuid4().hex[:16]}"

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=str(decision.get("approver_id", ctx.requester_id)),
                requested_by=ctx.requester_id,
                approved_by=str(decision.get("approver_id", "")),
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", hooks.release_version),
                event_type="authz_check",
                decision="deny" if res.decision == PolicyDecision.DENY else "edited",
                step_or_op_key=f"revalidate:{new_op_id}",
                attempt=wf_deps.attempt,
                operation_id=new_op_id,
                details={
                    "reason": res.reason,
                    "edited_action": edited_plan.action,
                    "amount_cents": edited_plan.amount_cents,
                },
            )
        )
        if res.decision == PolicyDecision.DENY:
            return {
                "change_plan": edited_plan.model_dump(mode="json"),
                "policy_decision": "deny",
                "error_category": "permission_denied",
            }
        return {
            "change_plan": edited_plan.model_dump(mode="json"),
            "policy_decision": "require_approval",
            "operation_id": new_op_id,
            "approved_plan_hash": edited_plan.plan_hash(),
        }

    def route_after_edit(state: IncidentDeskState) -> str:
        if state.get("policy_decision") == "deny":
            return "finalize_denied"
        return "execute_change"

    # Node 8: execute_change (Approved-hash check + Idempotent write + Timeout reconciliation)
    async def execute_change(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id")
        approved_hash = state.get("approved_plan_hash")
        decision = state.get("approval_decision") or {}
        approver_id = str(decision.get("approver_id") or "")

        assert plan is not None and operation_id is not None

        if not approved_hash or plan.plan_hash() != approved_hash:
            await _record_audit(
                AuditEvent(
                    tenant_id=ctx.tenant_id,
                    actor_id=approver_id or ctx.requester_id,
                    requested_by=ctx.requester_id,
                    approved_by=approver_id or None,
                    thread_id=ctx.thread_id,
                    run_id=ctx.run_id,
                    workflow_version=state.get("release_version", hooks.release_version),
                    event_type="authz_check",
                    decision="deny",
                    step_or_op_key=f"hash_mismatch:{operation_id}",
                    attempt=wf_deps.attempt,
                    operation_id=operation_id,
                    details={
                        "reason": "approved_plan_hash mismatch",
                        "expected": approved_hash,
                        "actual": plan.plan_hash(),
                    },
                )
            )
            return {
                "error_category": "permission_denied",
                "final_status": "denied_plan_mismatch",
            }

        write_spec = hooks.resolve_write_spec(executor, ctx, plan, operation_id)
        try:
            result = await executor.invoke_write(
                ctx,
                spec=write_spec,
                approved_by=approver_id,
                operation_id=operation_id,
                simulate_timeout=wf_deps.simulate_write_timeout,
            )
            maybe_inject_fault("after_write")
        except TimeoutError:
            await _record_audit(
                AuditEvent(
                    tenant_id=ctx.tenant_id,
                    actor_id=approver_id or ctx.requester_id,
                    requested_by=ctx.requester_id,
                    approved_by=approver_id or None,
                    thread_id=ctx.thread_id,
                    run_id=ctx.run_id,
                    workflow_version=state.get("release_version", hooks.release_version),
                    event_type="tool_write",
                    tool_id=write_spec.tool_id,
                    operation_id=operation_id,
                    decision="outcome_unknown",
                    step_or_op_key=f"write:{operation_id}",
                    attempt=wf_deps.attempt,
                    details={"reason": "timeout; reconciling"},
                )
            )
            reconciled = await executor.reconcile_operation(
                ctx,
                mcp_url=write_spec.reconcile_mcp_url,
                operation_id=operation_id,
                approved_by=approver_id,
            )
            rec_decision = (
                "succeeded" if reconciled.get("status") == "succeeded" else "outcome_unknown"
            )
            await _record_audit(
                AuditEvent(
                    tenant_id=ctx.tenant_id,
                    actor_id=approver_id or ctx.requester_id,
                    requested_by=ctx.requester_id,
                    approved_by=approver_id or None,
                    thread_id=ctx.thread_id,
                    run_id=ctx.run_id,
                    workflow_version=state.get("release_version", hooks.release_version),
                    event_type="reconciliation",
                    tool_id="get_operation_status",
                    operation_id=operation_id,
                    decision=rec_decision,
                    step_or_op_key=f"reconcile:{operation_id}",
                    attempt=wf_deps.attempt,
                    details=reconciled,
                )
            )
            if reconciled.get("status") == "succeeded":
                return {"execution_result": reconciled["result_payload"]}
            return {
                "error_category": "outcome_unknown",
                "final_status": "outcome_unknown",
                "execution_result": reconciled,
            }

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=approver_id or ctx.requester_id,
                requested_by=ctx.requester_id,
                approved_by=approver_id or None,
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", hooks.release_version),
                event_type="tool_write",
                tool_id=write_spec.tool_id,
                operation_id=operation_id,
                decision="succeeded",
                step_or_op_key=f"write:{operation_id}",
                attempt=wf_deps.attempt,
                details=result,
            )
        )
        return {"execution_result": result}

    def route_after_execute(state: IncidentDeskState) -> str:
        if state.get("error_category") is not None:
            return END
        return "verify_end_state"

    # Node 9: verify_end_state (Deterministic read-back + cross-thread memory persistence)
    async def verify_end_state(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id") or ctx.run_id
        exec_res = state.get("execution_result") or {}
        assert plan is not None

        try:
            passed = await hooks.verify_domain_end_state(
                executor, ctx, plan, operation_id, exec_res
            )
        except TimeoutError:
            passed = False
        if passed and runtime.store is not None:
            await runtime.store.aput(
                (ctx.tenant_id, "memory", hooks.workflow_name),
                operation_id,
                hooks.build_memory_payload(ctx, plan, operation_id),
            )
        return {
            "verification_passed": passed,
            "final_status": "completed" if passed else "verification_failed",
        }

    def finalize_no_action(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "no_action", "verification_passed": True}

    def finalize_denied(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "denied", "verification_passed": False}

    def finalize_rejected(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "rejected", "verification_passed": False}

    builder = StateGraph(IncidentDeskState, context_schema=context_schema)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("investigate", investigate)
    builder.add_node("propose_plan", propose_plan)
    builder.add_node("finalize_no_action", finalize_no_action)
    builder.add_node("evaluate_policy", evaluate_policy)
    builder.add_node("finalize_denied", finalize_denied)
    builder.add_node("record_approval_request", record_approval_request)
    builder.add_node("await_approval", await_approval)
    builder.add_node("finalize_rejected", finalize_rejected)
    builder.add_node("revalidate_edit", revalidate_edit)
    builder.add_node("execute_change", execute_change)
    builder.add_node("verify_end_state", verify_end_state)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "investigate")
    builder.add_edge("investigate", "propose_plan")
    builder.add_conditional_edges(
        "propose_plan",
        route_after_plan,
        {"finalize_no_action": "finalize_no_action", "evaluate_policy": "evaluate_policy"},
    )
    builder.add_edge("finalize_no_action", END)
    builder.add_conditional_edges(
        "evaluate_policy",
        route_after_policy,
        {
            "finalize_denied": "finalize_denied",
            "record_approval_request": "record_approval_request",
        },
    )
    builder.add_edge("finalize_denied", END)
    builder.add_edge("record_approval_request", "await_approval")
    builder.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {
            "execute_change": "execute_change",
            "revalidate_edit": "revalidate_edit",
            "finalize_rejected": "finalize_rejected",
        },
    )
    builder.add_edge("finalize_rejected", END)
    builder.add_conditional_edges(
        "revalidate_edit",
        route_after_edit,
        {"finalize_denied": "finalize_denied", "execute_change": "execute_change"},
    )
    builder.add_conditional_edges(
        "execute_change",
        route_after_execute,
        {"verify_end_state": "verify_end_state", END: END},
    )
    builder.add_edge("verify_end_state", END)

    return builder.compile(checkpointer=checkpointer, store=store)
